from utilities import *

if __name__ == "__main__":

    if os.path.exists(OP1_DIR):
        shutil.rmtree(OP1_DIR)
    os.makedirs(OP1_DIR)
    os.makedirs(OP1_DIR+"/reconstructed_images")

    images_right     = load_images('./img2/')
    images_left     = load_images('./img3/')
    poses = get_poses('./poses.txt') 

    color_map = extract_color(images_left)

    parallax_map, disp = create_parallax_map(images_left, images_right)
    
    B_matrix = get_baseline_matrix(B, K)

    Point_Cloud, Point_Cloud_Colors = get_point_cloud(B_matrix, parallax_map, color_map, poses)

    image_point_cloud(Point_Cloud, Point_Cloud_Colors, poses, images_right[0].shape[1], images_right[0].shape[0])
    print('Outputs saved in output_question_1 folder ')