import cv2
import numpy as np

class DonutCorners():
    rot90 = np.array([[0, -1], [1, 0]])

    # pylint: disable=too-many-instance-attributes
    def __init__(self, image, **kwargs):
        if isinstance(image, str):
            self.src = cv2.imread(image)
        else:
            self.src = image

        # sobel params
        self.ksize = 3
        self.scale = 1
        self.delta = 0
        self.ddepth = cv2.CV_8U

        # donut params
        self.radii = [10, 20, 40]
        self.round = False
        self.nearest = True

        # grid params
        self.grid_size = 10

        self.__dict__.update(kwargs)

        self.preprocess()

    def preprocess(self):

        edges_x = cv2.Sobel(self.src, self.ddepth, dx=1, dy=0, ksize=self.ksize,
                            scale=self.scale, delta=self.delta, borderType=cv2.BORDER_DEFAULT)
        edges_y = cv2.Sobel(self.src, self.ddepth, dx=0, dy=1, ksize=self.ksize,
                            scale=self.scale, delta=self.delta, borderType=cv2.BORDER_DEFAULT)

        edges_x_max = np.max(edges_x, axis=-1, keepdims=True)
        edges_y_max = np.max(edges_y, axis=-1, keepdims=True)
        self.slopes = np.append(edges_x_max, edges_y_max, axis=-1)


    def score_point(self, point):
        dozen = [self.bake_donut(point, radius) for radius in self.radii]
        return sum(dozen)


    def bake_donut(self, point, radius):
        # identify points of interest
        pois = self.donut(point, radius)

        # trace rays from each point
        rays = [self.raytrace(p, point) for p in pois]

        # find the strength of each ray
        strengths = [DonutCorners.score_ray(ray) for ray in rays]

        return sum(strengths[:4])


    def donut(self, point, radius):
        pois = []
        return pois


    def raytrace(self, p_1, p_2):
        floor = np.floor
        rnd = np.round
        uv = p_2 - p_1
        l = np.linalg.norm(uv)
        uv = uv/l
        perp = DonutCorners.rot90.dot(uv)

        if self.nearest:
            return [perp.dot(self.slopes[tuple(p_1 + floor(uv*i))]) for i in range(rnd(l))]
        
        # interpolate here
        return None


    @classmethod
    def score_ray(cls, profile):
        pass


if __name__ == "__main__":
    img = cv2.imread('images/bldg-1.jpg')
    #crop
    img = img[500:1000, 500:1000]
    dc = DonutCorners(img)
    
    show = dc.slopes[...,1]
    print(show.shape)
    cv2.imshow('Display',show)
    cv2.waitKey(30000)