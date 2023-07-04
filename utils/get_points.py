import time
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import open3d as o3d

intrinsics = np.array([
            [605.7855224609375, 0., 324.2651672363281, 0.0],
            [0., 605.4981689453125, 238.91090393066406, 0.0],
            [0., 0., 1., 0.0],
            [0., 0., 0., 1.],])

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def launch_realsense(pixel_width, pixel_high, fps, found_rgb=False):
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("Can't launch rgb camera")
        exit(0)

    config.enable_stream(rs.stream.depth, pixel_width, pixel_high, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, pixel_width, pixel_high, rs.format.bgr8, fps)

    align_to = rs.stream.color
    alignedFs = rs.align(align_to)
    # Start streaming
    pipeline.start(config)

    # Create folders by date
    save_path = os.path.join(os.getcwd(), "out_data",
                             time.strftime("%Y_%m_%d_%H_%M_%S",
                                           time.localtime()))
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, "rgb"))
    os.makedirs(os.path.join(save_path, "depth"))
    os.makedirs(os.path.join(save_path, "depth_colormap"))

    # cv2.namedWindow("camera in real time", cv2.WINDOW_AUTOSIZE)
    # saved_color_image = None
    # saved_depth_mapped_image = None
    try:
        flag = 0
        while True:
            if flag == 0:
                time.sleep(2)
                flag = 1
                continue
            # Wait for a coherent pair of frames: rgb and depth
            frames = pipeline.wait_for_frames()
            align_frames = alignedFs.process(frames)
            depth_frame = align_frames.get_depth_frame()
            color_frame = align_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_profile = color_frame.get_profile()
            cvsprofile = rs.video_stream_profile(color_profile)
            color_intrin = cvsprofile.get_intrinsics()
            color_intrin_part = [color_intrin.ppx, color_intrin.ppy,
                                 color_intrin.fx, color_intrin.fy]
            print('**color_intrin_part**:',color_intrin_part)
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)

            # depth_colormap_dim = depth_colormap.shape
            # color_colormap_dim = color_image.shape
            #
            # if depth_colormap_dim != color_colormap_dim:
            #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
            #                                      interpolation=cv2.INTER_AREA)
            #     images = np.hstack((resized_color_image, depth_colormap))
            # else:
            #     images = np.hstack((color_image, depth_colormap))

            # # Show images
            # cv2.imshow("camera in real time", images)
            # key = cv2.waitKey(1)

            # Save the image
            # if key & 0xFF == ord('s'):
            saved_count = 0
            for filename in os.listdir(os.path.join((save_path), "rgb")):
                if filename.endswith('.png'):
                    saved_count += 1
            print('save data:',saved_count)
            saved_color_image = color_image
            saved_depth_image = depth_image
            saved_depth_mapped_image = depth_colormap
            # save rgb png
            cv2.imwrite(os.path.join((save_path), "rgb",
                                     "rgb_{}.png".format(saved_count)),saved_color_image)
            # save depth_colormap png
            cv2.imwrite(os.path.join((save_path), "depth_colormap",
                                     "depth_colormap_{}.png".format(saved_count)),
                        saved_depth_mapped_image)
            # save depth png
            cv2.imwrite(os.path.join((save_path), "depth",
                                     "depth_{}.png".format(saved_count)),
                        saved_depth_image)
            # save depth npy
            np.save(os.path.join((save_path), "depth",
                                 "depth_{}.npy".format(saved_count)), saved_depth_image)

            depth_path = os.path.join((save_path), "depth", "depth_{}.npy".format(saved_count))
            color_path = os.path.join((save_path), "rgb", "rgb_{}.png".format(saved_count))
            return depth_path, color_path

    finally:
        # Stop streaming
        pipeline.stop()


def loadRGB(color_file):
    return cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)

def loadDepth(depth_file):
    return cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

def save_points(depth_path, color_path):

    colors = loadRGB(color_path).astype(np.float32) / 255.0
    depths = np.load(depth_path) # loadDepth(depth_path)

    # convert RGB-D to point cloud
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # depth factor
    s = 1000.0
    xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points.reshape((-1, 3))
    colors = colors.reshape((-1, 3))

    mask = np.where(points[:, 2] < 1)
    points = points[mask]
    colors = colors[mask]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cloud, coord])


    base_dir = os.path.dirname(os.path.dirname(color_path))
    points_file = os.path.join(base_dir, 'points.npy')
    colors_file = os.path.join(base_dir, 'colors.npy')
    np.save(points_file, points)
    np.save(colors_file, colors)
    return points_file, colors_file

if __name__ == '__main__':
    depth_path, color_path = launch_realsense(pixel_width=640, pixel_high=480, fps=30)
    save_points(depth_path, color_path)
