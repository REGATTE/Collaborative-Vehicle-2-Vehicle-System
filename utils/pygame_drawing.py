"""import pygame
import numpy as np
import pdb

class PyGameDrawing(object):
    def __init__(self):
        pass

    @staticmethod
    def draw_bbox_in_pygame(surface, boxes, color=(0, 255, 0), thickness=2):
        if boxes is None or len(boxes) == 0:
                return

        for box in boxes:
            if box is None:
                continue # Handle NoneType

            # Convert coordinates to integers
            points = box.astype(np.int32)
            
            if points.shape[0]<8:
                continue

            # Draw front face
            pygame.draw.line(surface, color, tuple(points[0]), tuple(points[1]), thickness)
            pygame.draw.line(surface, color, tuple(points[1]), tuple(points[2]), thickness)
            pygame.draw.line(surface, color, tuple(points[2]), tuple(points[3]), thickness)
            pygame.draw.line(surface, color, tuple(points[3]), tuple(points[0]), thickness)
            
            # Draw back face
            pygame.draw.line(surface, color, tuple(points[4]), tuple(points[5]), thickness)
            pygame.draw.line(surface, color, tuple(points[5]), tuple(points[6]), thickness)
            pygame.draw.line(surface, color, tuple(points[6]), tuple(points[7]), thickness)
            pygame.draw.line(surface, color, tuple(points[7]), tuple(points[4]), thickness)
            
            # Draw lines connecting front and back faces
            pygame.draw.line(surface, color, tuple(points[0]), tuple(points[4]), thickness)
            pygame.draw.line(surface, color, tuple(points[1]), tuple(points[5]), thickness)
            pygame.draw.line(surface, color, tuple(points[2]), tuple(points[6]), thickness)
            pygame.draw.line(surface, color, tuple(points[3]), tuple(points[7]), thickness)

"""

import pygame
import numpy as np

class PyGameDrawing(object):
    def __init__(self):
        pass

    @staticmethod
    def draw_bbox_in_pygame(surface, boxes, color=(0, 255, 0), thickness=2):
        if boxes is None or len(boxes) == 0:
            return

        for box in boxes:
            if box is None:
                continue  # Handle NoneType

            # Convert coordinates to integers
            points = box.astype(np.int32)

            # Ensure the box has exactly 8 points
            if points.shape[0] < 8:
                print("Skipping box, insufficient points:", points)
                continue

            # Draw front face
            pygame.draw.line(surface, color, tuple(points[0]), tuple(points[1]), thickness)
            pygame.draw.line(surface, color, tuple(points[1]), tuple(points[2]), thickness)
            pygame.draw.line(surface, color, tuple(points[2]), tuple(points[3]), thickness)
            pygame.draw.line(surface, color, tuple(points[3]), tuple(points[0]), thickness)
            
            # Draw back face
            pygame.draw.line(surface, color, tuple(points[4]), tuple(points[5]), thickness)
            pygame.draw.line(surface, color, tuple(points[5]), tuple(points[6]), thickness)
            pygame.draw.line(surface, color, tuple(points[6]), tuple(points[7]), thickness)
            pygame.draw.line(surface, color, tuple(points[7]), tuple(points[4]), thickness)
            
            # Draw lines connecting front and back faces
            pygame.draw.line(surface, color, tuple(points[0]), tuple(points[4]), thickness)
            pygame.draw.line(surface, color, tuple(points[1]), tuple(points[5]), thickness)
            pygame.draw.line(surface, color, tuple(points[2]), tuple(points[6]), thickness)
            pygame.draw.line(surface, color, tuple(points[3]), tuple(points[7]), thickness)

            # Debug: Draw small circles at each corner of the bounding box
            for i, point in enumerate(points):
                if 0 <= point[0] < surface.get_width() and 0 <= point[1] < surface.get_height():
                    pygame.draw.circle(surface, (255, 0, 0), (point[0], point[1]), 3)  # Red dots at each corner
                else:
                    print(f"Point {i} out of screen bounds: {point}")
