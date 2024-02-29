# -*- coding: UTF-8 -*-
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyrotation.conversion import matrix_from_quaternion, matrix_from_euler_angle, fill_matrix


# File    ：visual.py
# Author  ：fzhiheng
# Date    ：2023/11/27

def plot_pose(fig, pose: np.ndarray, name: str, row=1, col=1, align_first=False, axis_length=8, inter=100, mark_start=True, mark_end=True, show_legend=True):
    """

    Args:
        pose: (n, 4, 4)
    Returns:

    """
    if align_first:
        pose = np.linalg.inv(pose[:1, :, :]) @ pose  # (n, 4, 4)

    x, y, z = pose[:, 0, 3], pose[:, 1, 3], pose[:, 2, 3]
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=name), row=row, col=col)
    if mark_start:
        fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', name=f"{name} start point"), row=row, col=col)
    if mark_end:
        fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers', name=f"{name} end point"), row=row, col=col)

    start_points = []
    x_axis_end = []
    y_axis_end = []
    z_axis_end = []
    for i, (R, t) in enumerate(zip(pose[:, :3, :3], pose[:, :3, 3])):
        if i % inter == 0:
            point = t
            x_end = R[:, 0] * axis_length + point
            y_end = R[:, 1] * axis_length + point
            z_end = R[:, 2] * axis_length + point

            start_points.append(point)
            x_axis_end.append(x_end)
            y_axis_end.append(y_end)
            z_axis_end.append(z_end)

    show_legend = show_legend
    for start_point, x_, y_, z_ in zip(start_points, x_axis_end, y_axis_end, z_axis_end):
        # 使用红色表示x轴
        fig.add_trace(go.Scatter3d(x=[start_point[0], x_[0]], y=[start_point[1], x_[1]], z=[start_point[2], x_[2]],
                                   mode='lines', name="x-axis", marker=dict(color='red'), showlegend=show_legend), row=row, col=col)
        # 使用绿色表示y轴
        fig.add_trace(go.Scatter3d(x=[start_point[0], y_[0]], y=[start_point[1], y_[1]], z=[start_point[2], y_[2]],
                                   mode='lines', name="y-axis", marker=dict(color='green'), showlegend=show_legend), row=row, col=col)
        # 使用蓝色表示z轴
        fig.add_trace(go.Scatter3d(x=[start_point[0], z_[0]], y=[start_point[1], z_[1]], z=[start_point[2], z_[2]],
                                   mode='lines', name="z-axis", marker=dict(color='blue'), showlegend=show_legend), row=row, col=col)
        show_legend = False


def plot_xyz(fig, xyz: np.ndarray, name: str, row=1, col=1, mark_start=True, mark_end=True):
    """

    Args:
        xyz: (n, 3)
    Returns:

    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=name), row=row, col=col)
    if mark_start:
        fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', name=f"{name} start point"), row=row, col=col)
    if mark_end:
        fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers', name=f"{name} end point"), row=row, col=col)


def plot_tum(fig, traj, name, row=1, col=1, align_first=False, axis_length=8, inter=100, mark_start=True, mark_end=True, show_legend=True):
    matrix = matrix_from_quaternion(np.roll(traj[:, 4:], 1, axis=-1))  # (n, 3, 3)
    full_matrix = fill_matrix(matrix, traj[:, 1:4])
    plot_pose(fig, full_matrix, name, row=row, col=col, align_first=align_first, axis_length=axis_length, inter=inter, mark_start=mark_start,
              mark_end=mark_end,
              show_legend=show_legend)


if __name__ == "__main__":
    num_poses = 1000

    # generate continuous rotation matrix
    x_angles = np.linspace(0, np.pi, num_poses)
    y_angles = np.linspace(0, np.pi, num_poses)
    z_angles = np.linspace(0, np.pi, num_poses)
    ypr = np.stack([y_angles, x_angles, z_angles], axis=1)
    matrix = matrix_from_euler_angle(ypr, axes=["z", "y", "x"])

    # generate continuous translation
    x = np.linspace(0, 100, num_poses)
    y = np.linspace(0, 20, num_poses)
    z = np.linspace(0, 2, num_poses)
    ts = np.stack([x, y, z], axis=1)

    # concatenate rotation and translation
    T = np.concatenate([matrix, ts[..., None]], axis=-1)
    T = np.concatenate([T, np.array([0, 0, 0, 1])[None, None, :].repeat(num_poses, axis=0)], axis=1)

    # plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    plot_pose(fig, T, "traj", row=1, col=1, align_first=True, axis_length=2, inter=20, mark_start=True)
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()
