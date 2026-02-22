# modules/avatar_3d.py

import plotly.graph_objects as go
import numpy as np
from typing import Optional, List, Tuple


# MediaPipe Pose connection pairs (indices)
POSE_CONNECTIONS = [
    # Torso
    (11, 12),  # Shoulders
    (11, 23),  # Left shoulder to left hip
    (12, 24),  # Right shoulder to right hip
    (23, 24),  # Hips
    # Left Arm
    (11, 13),  # Left shoulder to left elbow
    (13, 15),  # Left elbow to left wrist
    # Right Arm
    (12, 14),  # Right shoulder to right elbow
    (14, 16),  # Right elbow to right wrist
    # Left Leg
    (23, 25),  # Left hip to left knee
    (25, 27),  # Left knee to left ankle
    # Right Leg
    (24, 26),  # Right hip to right knee
    (26, 28),  # Right knee to right ankle
    # Head
    (0, 11),   # Nose to left shoulder (approximation)
    (0, 12),   # Nose to right shoulder
]

# Face mesh simplified connections (key landmarks only)
FACE_CONNECTIONS_SIMPLE = [
    # Jaw line (simplified)
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10),
    # Left eye
    (33, 160), (160, 158), (158, 133),
    (133, 153), (153, 144), (144, 33),
    # Right eye
    (362, 385), (385, 387), (387, 263),
    (263, 373), (373, 380), (380, 362),
    # Mouth
    (61, 146), (146, 91), (91, 181), (181, 84),
    (84, 17), (17, 314), (314, 405), (405, 321),
    (321, 375), (375, 291), (291, 61),
]


def create_pose_avatar_3d(
    landmarks_3d: np.ndarray,
    title: str = "3D Body Avatar",
    movement_flags: Optional[dict] = None
) -> go.Figure:
    """
    Create an interactive 3D stick-figure avatar from
    MediaPipe Pose landmarks.

    Args:
        landmarks_3d: Array of shape (33, 4) with x, y, z, visibility
        title: Plot title
        movement_flags: Dict with 'is_rocking', 'is_flapping' etc.

    Returns:
        Plotly Figure object
    """
    x = landmarks_3d[:, 0]
    y = -landmarks_3d[:, 1]  # Flip Y for natural orientation
    z = landmarks_3d[:, 2]
    visibility = landmarks_3d[:, 3]

    # Joint points
    joint_colors = ['blue'] * len(x)

    # Highlight active joints
    if movement_flags:
        if movement_flags.get('is_hand_flapping', False):
            # Highlight wrists
            for idx in [15, 16]:
                joint_colors[idx] = 'red'
        if movement_flags.get('is_rocking', False):
            # Highlight torso
            for idx in [11, 12, 23, 24]:
                joint_colors[idx] = 'orange'

    # Create joint scatter
    joints = go.Scatter3d(
        x=x, y=z, z=y,
        mode='markers',
        marker=dict(
            size=5,
            color=joint_colors,
            opacity=0.9
        ),
        name='Joints',
        hovertext=[
            f"Joint {i} (v={visibility[i]:.2f})"
            for i in range(len(x))
        ]
    )

    # Create bone lines
    bone_x, bone_y, bone_z = [], [], []
    bone_colors = []

    for start, end in POSE_CONNECTIONS:
        if (
            start < len(x) and end < len(x)
            and visibility[start] > 0.3
            and visibility[end] > 0.3
        ):
            bone_x.extend([x[start], x[end], None])
            bone_y.extend([z[start], z[end], None])
            bone_z.extend([y[start], y[end], None])

    bones = go.Scatter3d(
        x=bone_x, y=bone_y, z=bone_z,
        mode='lines',
        line=dict(color='cyan', width=4),
        name='Skeleton',
        hoverinfo='skip'
    )

    # Create figure
    fig = go.Figure(data=[joints, bones])

    # Status text
    status_text = ""
    if movement_flags:
        if movement_flags.get('is_rocking'):
            status_text += "⚠️ ROCKING DETECTED  "
        if movement_flags.get('is_hand_flapping'):
            status_text += "⚠️ HAND FLAPPING  "
        if not status_text:
            status_text = "✅ Normal Movement"

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{status_text}</sub>",
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(
                range=[-0.2, 1.2], showgrid=False,
                zeroline=False, showticklabels=False,
                title=''
            ),
            yaxis=dict(
                range=[-1, 1], showgrid=False,
                zeroline=False, showticklabels=False,
                title=''
            ),
            zaxis=dict(
                range=[-1.2, 0.2], showgrid=False,
                zeroline=False, showticklabels=False,
                title=''
            ),
            bgcolor='rgb(10, 10, 30)',
            camera=dict(
                eye=dict(x=0, y=-2.0, z=0.3)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5)
        ),
        paper_bgcolor='rgb(10, 10, 30)',
        plot_bgcolor='rgb(10, 10, 30)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
        showlegend=False
    )

    return fig


def create_face_avatar_3d(
    landmarks_3d: np.ndarray,
    gaze_direction: str = "center",
    expression: str = "neutral"
) -> go.Figure:
    """
    Create a 3D face mesh visualization from
    MediaPipe Face Mesh landmarks.
    """
    x = landmarks_3d[:, 0]
    y = -landmarks_3d[:, 1]
    z = landmarks_3d[:, 2]

    # Determine point colors based on regions
    colors = ['lightblue'] * len(x)

    # Eye regions
    left_eye_idx = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]
    mouth_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    for i in left_eye_idx + right_eye_idx:
        if i < len(colors):
            colors[i] = 'lime'
    for i in mouth_idx:
        if i < len(colors):
            colors[i] = 'salmon'

    # Face points
    points = go.Scatter3d(
        x=x, y=z, z=y,
        mode='markers',
        marker=dict(size=1.5, color=colors, opacity=0.7),
        name='Face Mesh',
        hoverinfo='skip'
    )

    # Face connections (simplified)
    line_x, line_y, line_z = [], [], []
    for start, end in FACE_CONNECTIONS_SIMPLE:
        if start < len(x) and end < len(x):
            line_x.extend([x[start], x[end], None])
            line_y.extend([z[start], z[end], None])
            line_z.extend([y[start], y[end], None])

    lines = go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color='rgba(100, 200, 255, 0.5)', width=1),
        name='Mesh Lines',
        hoverinfo='skip'
    )

    fig = go.Figure(data=[points, lines])

    gaze_color = "🟢" if gaze_direction == "center" else "🔴"

    fig.update_layout(
        title=dict(
            text=(
                f"3D Face Model<br>"
                f"<sub>{gaze_color} Gaze: {gaze_direction} "
                f"| Expression: {expression}</sub>"
            ),
            font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(
                showgrid=False, zeroline=False,
                showticklabels=False, title=''
            ),
            yaxis=dict(
                showgrid=False, zeroline=False,
                showticklabels=False, title=''
            ),
            zaxis=dict(
                showgrid=False, zeroline=False,
                showticklabels=False, title=''
            ),
            bgcolor='rgb(10, 10, 30)',
            camera=dict(eye=dict(x=0, y=-1.5, z=0.1)),
            aspectmode='data'
        ),
        paper_bgcolor='rgb(10, 10, 30)',
        plot_bgcolor='rgb(10, 10, 30)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=50, b=0),
        height=400,
        showlegend=False
    )

    return fig