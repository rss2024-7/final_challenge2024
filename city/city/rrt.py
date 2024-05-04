import numpy as np
import random
import math


class RRT:
    def __init__(self, map_info):
        self.max_distance = 2 / 0.054 
        self.dist_thres = 1 / 0.054
        self.step_size = 1.5 / 0.054

        self.map_data = np.array(map_info[0]) # AKA occupancy grid
        self.map_width = map_info[1] #1730
        self.map_height = map_info[2] #1300
        self.map_resolution = map_info[3]

        # self.hall_pixels = np.where(self.map_data == 0)[0] # the places that are beige
        self.hall_pixels = []
        for i in range(len(self.map_data)):
            if self.map_data[i] == 0:
                self.hall_pixels.append(i)

        # self.wall_pixels = np.where(self.map_data == 100)[0]
        self.nodes = []


        # TO-DO: make less jank
        x = 25.900000
        y = 48.50000
        theta = 3.14
        resolution = 0.0504
        self.transform = np.array([[np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0,0,1]])

        self.x_bounds = (-61.3, 25.9)
        self.y_bounds = (-16.2, 48.5)

        self.none_visited = np.full(self.map_data.shape, False)
        self.no_parents = np.full(self.map_data.shape, None)

        self.freespace = np.full(self.map_data.shape, False)
        self.freespace[np.where((self.map_data == 0) | (self.map_data == 25) | (self.map_data == 75))] = True

        self.row_nums = np.indices(self.map_data.shape)[0] // self.map_width
        self.col_nums = np.indices(self.map_data.shape)[0] % self.map_width

        self.zero_angles = np.full(self.map_data.shape, 0.0)

    def sample_free_index(self, visited):
        """ Samples random map index that is unvisited and is free space. """
        sample_space_indices = np.where(self.freespace & ~visited)[0]
        return np.random.choice(sample_space_indices)
    
    def get_nearest_index(self, visited, index):
        graph_indices = np.where(visited)[0]

        row_distances = self.row_nums[graph_indices] - self.row_nums[index]
        col_distances = self.col_nums[graph_indices] - self.col_nums[index]
        distances = row_distances ** 2 + col_distances ** 2
        return graph_indices[np.where(distances == np.min(distances))][0]
    
    def plan_straight_line(self, start, end):
        start = np.array([start[0], start[1]]) #[x,y]
        end = np.array([end[0],end[1]]) #[x,y]

        start_pixel = self.real_to_pixel(start)
        end_pixel = self.real_to_pixel(end)

        start_pixel_i, start_pixel_j = start_pixel
        end_pixel_i, end_pixel_j = end_pixel

        start_index = self.index_from_pixel(start_pixel)
        end_index = self.index_from_pixel(end_pixel)

        # check for straight line path to goal
        intersected_i, intersected_j = self.get_intersected_indices(start_pixel_i, start_pixel_j,
                            end_pixel_i, end_pixel_j)
        intersected_indices = intersected_i * self.map_width + intersected_j
        collisions = ~self.freespace[intersected_indices]
        if not np.any(collisions):
            path = np.array([start_index, end_index])
            pixels = np.zeros((len(path), 2))
            pixels[:, 0] = path // self.map_width
            pixels[:, 1] = path % self.map_width
            points = self.pixels_to_real(pixels)
            return points

        return None


    def plan_path(self, start, end):
        start = np.array([start[0], start[1]]) #[x,y]
        end = np.array([end[0],end[1]]) #[x,y]

        start_pixel = self.real_to_pixel(start)
        end_pixel = self.real_to_pixel(end)

        start_pixel_i, start_pixel_j = start_pixel
        end_pixel_i, end_pixel_j = end_pixel

        start_index = self.index_from_pixel(start_pixel)
        end_index = self.index_from_pixel(end_pixel)

        if not self.freespace[end_index]:
            return None

        visited = np.copy(self.none_visited)
        visited[start_index] = True

        angles = np.copy(self.zero_angles)
        # angles[start_index] = start_theta 

        parents = np.copy(self.no_parents)

        for _ in range(self.map_data.shape[0]):
            # sample q_rand from C_free
            random_index = end_index if np.random.random() < 0.05 else self.sample_free_index(visited)
            # random_index = np.random.randint(0, self.map_data.shape[0])

            # get q_near
            nearest_index = self.get_nearest_index(visited, random_index)
            nearest_i = self.row_nums[nearest_index]
            nearest_j = self.col_nums[nearest_index]

            # check for straight line path to goal
            intersected_i, intersected_j = self.get_intersected_indices(nearest_i, nearest_j,
                                end_pixel_i, end_pixel_j)
            intersected_indices = intersected_i * self.map_width + intersected_j
            collisions = ~self.freespace[intersected_indices]
            if not np.any(collisions):
                parents[end_index] = nearest_index
                visited[end_index] = True
                
                current = end_index
                path = []
                while current is not None:
                    path.append(current)
                    current = parents[current]
                path.append(start_index)
                path.reverse()

                path = np.array(path)
                pixels = np.zeros((len(path), 2))
                pixels[:, 0] = path // self.map_width
                pixels[:, 1] = path % self.map_width
                points = self.pixels_to_real(pixels)
                return points


            # finding q_new
            intersected_i, intersected_j = self.get_intersected_indices(nearest_i, nearest_j,
                                self.row_nums[random_index], self.col_nums[random_index])

            intersected_indices = intersected_i * self.map_width + intersected_j
            collisions = ~self.freespace[intersected_indices]
            
            # only look at pixels before collision
            if np.any(collisions):
                first_collision_index = np.argmax(collisions)
                intersected_indices = intersected_indices[:first_collision_index]

            # only look at unvisited pixels
            intersected_indices = intersected_indices[np.where(~visited[intersected_indices])]

            if (len(intersected_indices) == 0): continue
            row_distances = self.row_nums[intersected_indices] - nearest_i
            col_distances = self.col_nums[intersected_indices] - nearest_j
            distances = row_distances ** 2 + col_distances ** 2

            distances = np.abs(distances - self.step_size**2)
            intersected_indices = intersected_indices[np.where(distances == np.min(distances))]

            if len(intersected_indices) == 0: continue

            new_index = intersected_indices[-1]

            # add new vertex and edge
            parents[new_index] = nearest_index
            visited[new_index] = True
            # angles[new_index] = angle

        return None


    def get_intersected_indices(self, start_i, start_j, end_i, end_j):
        # Calculate the number of steps needed in each dimension
        steps = max(abs(end_i - start_i), abs(end_j - start_j)) + 1
        
        # Generate the indices along the line
        indices_i = np.linspace(start_i, end_i, steps, dtype=int)
        indices_j = np.linspace(start_j, end_j, steps, dtype=int)
        
        return indices_i, indices_j
            


    # utils 
    def pixels_to_real(self, pixels):
        path = []
        for pixel in pixels:
            path.append(self.pixel_to_real(pixel))
        return path
        pixels *= self.map_resolution
        num_pixels = pixels.shape[0]
        pixels = np.hstack((pixels, np.ones((num_pixels, 1)))).reshape((3, num_pixels))
        points = np.linalg.inv(self.transform) @ pixels
        points = points.reshape((num_pixels, 3))

        return points[:, :2]

    def pixel_to_real(self, pix, column_row=False):
        # if column_row = True, pixel format is [v (column index), u (row index)] 
        if column_row:
            pixel = np.array([pix[0], pix[1]]) 
        else:
            pixel = np.array([pix[1], pix[0]]) 

        pixel = pixel * self.map_resolution
        pixel = np.array([*pixel,1.0])
        pixel = np.linalg.inv(self.transform) @ pixel
        point = pixel

        # returns real x,y 
        return np.array([point[0], point[1]])

    def real_to_pixel(self, pt, column_row=False):
        #takes in [x,y] real coordinate
        point = np.array([pt[0], pt[1], 1.0])
        point = self.transform @ point
        point = point / self.map_resolution
        pixel = np.floor(point)

        if column_row: # returns [v (column index), u (row index)] 
            return np.array([pixel[0], pixel[1]])
        else:
            # returns [u (row index), v (column index)]
            return np.array([pixel[1], pixel[0]])
        
    def real_to_pixels(self, pts):
        #takes in [x,y] real coordinate
        points = np.hstack((pts, np.ones(pts.shape[0], 1)))
        points = self.transform @ points
        points = points / self.map_resolution
        pixels = np.floor(points)

        pixels[:, [0, 1]] = pixels[:, [1, 0]]

        return pixels
    
    def indices_from_pixels(self, pixels):
        return (pixels[:, 0] * self.map_width + pixels[:, 1]).astype(int)

    
    def index_from_pixel(self, pixel, column_row=False):
        # calculate the index of the row-major map
        if column_row: # pixel = [v, u]
            return int(pixel[1] * self.map_width + pixel[0])
        else: # pixel = [u, v]
            return int(pixel[0] * self.map_width + pixel[1])
    
    def pixel_from_index(self, index, column_row=False):
        # calculate the index of the row-major map
        u = int(index/self.map_width)
        v = index - u * self.map_width
        if column_row: # pixel = [v, u]
            return np.array([v,u])
        else: # pixel = [u, v]
            return np.array([u,v])
