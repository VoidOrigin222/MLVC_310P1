#!/usr/bin/env python3
"""Check frame IDs, alternating target/blank frames, and source-space box stability."""

import argparse
from collections import Counter
import json
from pathlib import Path


def iou(first, second):
    width = max(0, min(first['cx'] + first['w'] / 2, second['cx'] + second['w'] / 2)
                - max(first['cx'] - first['w'] / 2, second['cx'] - second['w'] / 2))
    height = max(0, min(first['cy'] + first['h'] / 2, second['cy'] + second['h'] / 2)
                 - max(first['cy'] - first['h'] / 2, second['cy'] - second['h'] / 2))
    intersection = width * height
    return intersection / (first['w'] * first['h'] + second['w'] * second['h'] - intersection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--alternate-blank', action='store_true')
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    frames = [json.loads(line) for line in args.report.read_text().splitlines()]
    if len(frames) != args.frames:
        raise ValueError('unexpected number of detection records')
    reference = json.loads(args.reference.read_text().splitlines()[0])['boxes'] if args.reference else None
    minimum_iou, maximum_score_error, count = 1.0, 0.0, 0
    for index, frame in enumerate(frames):
        if frame['frame_id'] != index:
            raise ValueError('frame ID is missing, duplicated or out of order')
        boxes = frame['boxes']
        blank = args.alternate_blank and index % 2
        expected = Counter() if blank else Counter({'bus': 1, 'person': 4})
        if Counter(box['label'] for box in boxes) != expected:
            raise ValueError('wrong source frame detections at frame %s' % index)
        count += len(boxes)
        if reference and not blank:
            remaining = list(boxes)
            for target in reference:
                candidates = [box for box in remaining if box['class_id'] == target['class_id']]
                if not candidates:
                    raise ValueError('reference category missing')
                best = max(candidates, key=lambda box: iou(box, target))
                minimum_iou = min(minimum_iou, iou(best, target))
                maximum_score_error = max(maximum_score_error, abs(best['conf'] - target['conf']))
                remaining.remove(best)
    if reference and (minimum_iou < 0.95 or maximum_score_error > 0.03):
        raise ValueError('detections changed beyond functional comparison tolerance')
    print(json.dumps({'frames': len(frames), 'boxes': count, 'frame_association': 'passed',
                      'minimum_reference_iou': minimum_iou if reference else None,
                      'maximum_confidence_difference': maximum_score_error if reference else None}, indent=2))


if __name__ == '__main__':
    main()
