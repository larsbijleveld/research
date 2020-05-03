# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf

# load an image
def load_image(filename, size=(1024,1024)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

extentionIn = ".png"
extentionOut = ".jpg"
i = 1
while i < 1098:

	# load source image
	#src_image = load_image('input/visual_',i,'.jpg')
	src_image = load_image("datasets/rect/rect/rect%d%s" % (i, extentionIn))
	print('Loaded', src_image.shape)
	# load model
	model = load_model('model_76790_highres.h5')
	# generate image from source
	gen_image = model.predict(src_image)
	# scale from [-1,1] to [0,1]
	gen_image = (gen_image + 1) / 2.0
	# plot the image
	pyplot.imshow(gen_image[0])
	pyplot.axis('off')
	# show title
	#pyplot.title('exerp0%d_model_76790_highres.h5' % (i + 24), fontsize="5.0")
	#pyplot.show()
	#plt.ioff()
	plt.savefig("results/mass_scifi/exerp0%d_model_76790_highres_mass.h5%s" % (i, extentionOut), bbox_inches='tight')
	print(i)
	K.clear_session()
	# no whitespace image
	# savefig('foo.png', bbox_inches='tight')
	i += 1
