import numpy as np

def points_inside_triangle(A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    xs=np.array((x1,x2,x3),dtype=float)
    ys=np.array((y1,y2,y3),dtype=float)

    # The possible range of coordinates that can be returned
    x_range=np.arange(np.min(xs),np.max(xs)+1)
    y_range=np.arange(np.min(ys),np.max(ys)+1)

    # Set the grid of coordinates on which the triangle lies. The centre of the
    # triangle serves as a criterion for what is inside or outside the triangle.
    X,Y=np.meshgrid( x_range,y_range )
    xc=np.mean(xs)
    yc=np.mean(ys)

    # From the array 'triangle', points that lie outside the triangle will be
    # set to 'False'.
    triangle = np.ones(X.shape,dtype=bool)
    for i in range(3):
        ii=(i+1)%3
        if xs[i]==xs[ii]:
            include = X *(xc-xs[i])/abs(xc-xs[i]) >= xs[i] *(xc-xs[i])/abs(xc-xs[i])
        else:
            poly=np.poly1d([(ys[ii]-ys[i])/(xs[ii]-xs[i]),ys[i]-xs[i]*(ys[ii]-ys[i])/(xs[ii]-xs[i])])
            include = Y *(yc-poly(xc))/abs(yc-poly(xc)) >= poly(X) *(yc-poly(xc))/abs(yc-poly(xc))
        triangle*=include

    # Output: array with the x- and y- coordinates of the points inside the
    # triangle.
    X = [int(x) for x in X[triangle]]
    Y = [int(y) for y in Y[triangle]]
    points = list(zip(X, Y))

    return points


def triangle_colour_average(points, img):
    total = 0 
    for p in points:
        total += img[p[1]][p[0]]
    try:
        average = total/len(points)
    except ZeroDivisionError:
        average = 0
        print('ZeroDivisionError')

    return average

def triangle_colourise(points, img, value):
    for p in points:
        img[p[1]][p[0]] = value

    return img
