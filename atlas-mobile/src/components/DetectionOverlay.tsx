/**
 * DetectionOverlay – renders bounding boxes on top of the camera preview.
 *
 * Pure presentational component: receives decoded detections + frame info,
 * maps model coordinates --> screen pixels, and draws coloured boxes with
 * labels.
 */
import { View, Text, StyleSheet, type ViewStyle } from 'react-native';
import {
  mapBoxToScreen,
  type Detection,
  type FrameInfo,
} from '../utils/tensor_decoder';
import { BOX_COLORS, MODEL_INPUT_SIZE } from '../theme';
import { Dimensions } from 'react-native';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

export interface DetectionOverlayProps {
  detections: Detection[];
  frameInfo: FrameInfo | null;
}

export function DetectionOverlay({
  detections,
  frameInfo,
}: DetectionOverlayProps) {
  if (detections.length === 0 || frameInfo == null) return null;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {detections.map((det, idx) => {
        const color = BOX_COLORS[det.classId % BOX_COLORS.length];
        const pct = Math.round(det.score * 100);

        const screenBox = mapBoxToScreen(
          det.box,
          frameInfo,
          SCREEN_W,
          SCREEN_H,
          MODEL_INPUT_SIZE,
        );

        const boxStyle: ViewStyle = {
          position: 'absolute',
          left: screenBox.x,
          top: screenBox.y,
          width: screenBox.width,
          height: screenBox.height,
          borderWidth: 2,
          borderColor: color,
          borderRadius: 4,
        };

        const labelAbove = screenBox.y > 22;

        return (
          <View key={`${det.label}-${idx}`} style={boxStyle}>
            <View
              style={[
                styles.labelBadge,
                { backgroundColor: color },
                labelAbove ? { top: -18, left: -2 } : { top: 2, left: 2 },
              ]}
            >
              <Text style={styles.labelText} numberOfLines={1}>
                {det.label} {pct}%
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  labelBadge: {
    position: 'absolute',
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: 3,
  },
  labelText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
});
