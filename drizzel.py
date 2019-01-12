import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.cbook as cbook


def get_image():
    delta = 0.25
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0) #center PSF
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1) #other PSF
    Z = Z2#Z2 - Z1  # difference of Gaussians
    return Z


def do_plot(ax, Z, transform):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   #extent=[-2, 4, -3, 2], clip_on=True)
                   extent=[-3, 3, -3, 3], clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)


# prepare image and figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
Z = get_image()

# image rotation
do_plot(ax1, Z, mtransforms.Affine2D())
do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(45))
#do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))

# image skew
#do_plot(ax2, Z, mtransforms.Affine2D().skew_deg(30, 15))
do_plot(ax2, Z, mtransforms.Affine2D().rotate_deg(45))

# scale and reflection
do_plot(ax3, Z, mtransforms.Affine2D().scale(-1, .5))

# everything and a translation
do_plot(ax4, Z, mtransforms.Affine2D().
        rotate_deg(30).skew_deg(30, 15).scale(-1, .5).translate(.5, -1))

plt.show()









def get_image():
    fn = cbook.get_sample_data('necked_tensile_specimen.png')
    arr = plt.imread(fn)
    # make background transparent
    # you won't have to do this if your car image already has a transparent background
    mask = (arr == (1,1,1,1)).all(axis=-1)
    arr[mask] = 0
    return arr

def imshow_affine(ax, z, *args, **kwargs):
    im = ax.imshow(z, *args, **kwargs)
    x1, x2, y1, y2 = im.get_extent()
    im._image_skew_coordinate = (x2, y1)
    return im

N = 7
x = np.linspace(0, 1, N)
y = x**1.1
heading = np.linspace(10, 90, N)
trajectory = list(zip(x, y, heading))
width, height = 0.3, 0.3
car = get_image()
fig, ax = plt.subplots()
for i, t in enumerate(trajectory, start=1):
    xi, yi, deg = t
    im = imshow_affine(ax, car, interpolation='none',
                       extent=[0, width, 0, height], clip_on=True,
                       alpha=0.8*i/len(trajectory))
    center_x, center_y = width//2, height//2
    im_trans = (mtransforms.Affine2D()
                .rotate_deg_around(center_x, center_y, deg)
                .translate(xi, yi)
                + ax.transData)
    im.set_transform(im_trans)

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.7)
plt.show()





#####################################################################
#def visualize_trajectory(self, trajectory=[[0,0,0,0], [0.1,0.1,0,0]]):
#    domain_fig = plt.figure()
#
#    for i, s in enumerate(trajectory):
#        x, y, speed, heading = s[:4]
#        car_xmin = x - self.REAR_WHEEL_RELATIVE_LOC
#        car_ymin = y - self.CAR_WIDTH / 2.
#
#        car_fig = matplotlib.patches.Rectangle(
#            [car_xmin,
#             car_ymin],
#            self.CAR_LENGTH,
#            self.CAR_WIDTH,
#            alpha=(0.8 * i) / len(trajectory) )
#        rotation = Affine2D().rotate_deg_around(
#            x, y, heading * 180 / np.pi) + plt.gca().transData
#        car_fig.set_transform(rotation)
#        plt.gca().add_patch(car_fig)
####################################################################


