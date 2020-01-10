# donut-corners
An experimental corner detection method.

Donut Corners is a corner detection method using the following srategy:

1. Calculate the sobel gradient of the image
2. Locate peaks in the sobel gradient
3. Pick a point to evaluate
4. Project rays from said point to surrounding peaks in the sobel gradient
5. Determine if said rays coincide with edges by averaging the component of the sobel gradient perpendicular to the ray
6. Take the n rays that most strongly coincide with edges while filtering out rays that are too close together
7. Calculate a 'corner score' for our point by averaging the strength of the strongest rays
8. Use a gradient ascent algorithm on the result from steps 3-7 to find local maxima of the 'corner score'. These local maxima are our corners.

## Project Objective
I wanted to create a novel method for detecting corners in images.

I wanted to use only cv2 and numpy.

I wanted to be done in under a week.

Oh, how mislead I was...


## Target Audience
Object detection is central to many fields including robotics, self driving cars, inventorying and more.

Phil Keller’s Lego part cataloguer might benefit from good corner detection.

I have a personal project that would benefit from good corner detection.

## Problems
1. The corner score algorithm is very computationally expensive.
2. I was unable to find a decent optimizer that will work for this project in the time alotted.
3. My homemade gradient ascent algorithm is pathetically inefficient.