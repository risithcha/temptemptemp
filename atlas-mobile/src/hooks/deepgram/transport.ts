/**
 * Deepgram Nova-3 streaming transport.
 *
 * Owns a single WebSocket to Deepgram's realtime listen endpoint and the raw
 * audio plumbing around it:
 *   - Lazy open (R9): the socket is opened only once the FIRST PCM frame
 *     arrives, using that frame's ACTUAL sample rate in the URL.  Android/Oboe
 *     often ignores the requested record rate, so we never hardcode it.
 *   - Auth via the `['token', <key>]` WebSocket subprotocol (RN-friendly; no
 *     custom headers needed).
 *   - Float32 [-1,1] -> Int16 LE conversion, sent as binary frames
 *     (encoding=linear16).
 *   - KeepAlive pings so the socket isn't dropped during silence.
 *   - Bounded pre-open buffering + exponential-backoff reconnect.
 */
import { DEEPGRAM_WS_BASE, DEEPGRAM_KEEPALIVE_MS } from '../../theme';
import type { DeepgramResultMessage } from './diarization';

export interface DeepgramTransportCallbacks {
  /** A parsed Deepgram "Results" message. */
  onMessage: (msg: DeepgramResultMessage) => void;
  /** Socket became OPEN (initial connect or after a reconnect). */
  onOpen?: () => void;
  /** Socket closed. */
  onClose?: (code: number, reason: string) => void;
  /** A transport-level error (socket error or reconnect exhaustion). */
  onError?: (error: Error) => void;
}

/** ~10s of 100ms frames buffered before the socket opens, then oldest dropped. */
const MAX_PENDING_FRAMES = 100;
const MAX_RECONNECT_ATTEMPTS = 5;
const WS_OPEN = 1;

export class DeepgramTransport {
  private readonly apiKey: string;
  private readonly callbacks: DeepgramTransportCallbacks;

  private ws: WebSocket | null = null;
  private sampleRate: number | null = null;
  private pending: ArrayBuffer[] = [];
  private keepAliveTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private userClosed = false;

  constructor(apiKey: string, callbacks: DeepgramTransportCallbacks) {
    this.apiKey = apiKey;
    this.callbacks = callbacks;
  }

  get isOpen(): boolean {
    return this.ws != null && this.ws.readyState === WS_OPEN;
  }

  /**
   * Feed one mono Float32 PCM frame.  The first call latches the sample rate
   * and lazily opens the socket; subsequent frames stream (or buffer until the
   * socket is OPEN).
   */
  pushPcm(samples: Float32Array, sampleRate: number): void {
    if (this.userClosed) return;

    if (this.sampleRate == null) {
      this.sampleRate = Math.round(sampleRate);
      this.open();
    }

    const frame = floatTo16BitPCM(samples);
    if (this.isOpen) {
      try {
        this.ws!.send(frame);
      } catch {
        this.bufferFrame(frame);
      }
    } else {
      this.bufferFrame(frame);
    }
  }

  /** Gracefully close: send CloseStream, stop timers, close socket. */
  close(): void {
    this.userClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopKeepAlive();
    this.pending = [];

    if (this.ws) {
      try {
        if (this.isOpen) {
          this.ws.send(JSON.stringify({ type: 'CloseStream' }));
        }
      } catch {
        /* ignore */
      }
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
  }

  private bufferFrame(frame: ArrayBuffer): void {
    this.pending.push(frame);
    if (this.pending.length > MAX_PENDING_FRAMES) {
      this.pending.shift();
    }
  }

  private open(): void {
    if (this.sampleRate == null || this.userClosed) return;

    const url = `${DEEPGRAM_WS_BASE}&sample_rate=${this.sampleRate}`;
    let ws: WebSocket;
    try {
      ws = new WebSocket(url, ['token', this.apiKey]);
    } catch (e) {
      this.callbacks.onError?.(e as Error);
      this.scheduleReconnect();
      return;
    }

    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    ws.onopen = () => {
      this.reconnectAttempts = 0;
      // Flush anything captured before the socket finished connecting.
      const queued = this.pending;
      this.pending = [];
      for (const frame of queued) {
        try {
          ws.send(frame);
        } catch {
          /* ignore */
        }
      }
      this.startKeepAlive();
      this.callbacks.onOpen?.();
    };

    ws.onmessage = (event: WebSocketMessageEvent) => {
      if (typeof event.data !== 'string') return;
      try {
        const msg = JSON.parse(event.data) as DeepgramResultMessage;
        if (msg.type === 'Results' || msg.channel != null) {
          this.callbacks.onMessage(msg);
        }
      } catch {
        /* non-JSON / keepalive acks – ignore */
      }
    };

    ws.onerror = (event: Event) => {
      const message = (event as { message?: string }).message;
      this.callbacks.onError?.(new Error(message ?? 'Deepgram socket error'));
    };

    ws.onclose = (event: WebSocketCloseEvent) => {
      this.stopKeepAlive();
      this.ws = null;
      this.callbacks.onClose?.(event?.code ?? 0, event?.reason ?? '');
      if (!this.userClosed) {
        this.scheduleReconnect();
      }
    };
  }

  private startKeepAlive(): void {
    this.stopKeepAlive();
    this.keepAliveTimer = setInterval(() => {
      if (this.isOpen) {
        try {
          this.ws!.send(JSON.stringify({ type: 'KeepAlive' }));
        } catch {
          /* ignore */
        }
      }
    }, DEEPGRAM_KEEPALIVE_MS);
  }

  private stopKeepAlive(): void {
    if (this.keepAliveTimer) {
      clearInterval(this.keepAliveTimer);
      this.keepAliveTimer = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.userClosed) return;
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.callbacks.onError?.(
        new Error('Deepgram: max reconnect attempts reached'),
      );
      return;
    }
    const delay = Math.min(8000, 500 * 2 ** this.reconnectAttempts);
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => this.open(), delay);
  }
}

/** Clamp Float32 [-1,1] to Int16 little-endian PCM and return its ArrayBuffer. */
export function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = input[i] < -1 ? -1 : input[i] > 1 ? 1 : input[i];
    out[i] = (s * 0x7fff) | 0;
  }
  return out.buffer;
}
