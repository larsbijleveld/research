# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib import pyplot as plt

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

# def limit_mem():
#	K.get_session().close()
#	cfg = K.tf.ConfigProto()
#	cfg.gpu_options.allow_growth = True
#	K.set_session(K.tf.Session(config=cfg))

extention = ".jpg"
i = 8
while i < 13:

	# load source image
	#src_image = load_image('input/visual_',i,'.jpg')
	src_image = load_image("input/visual_%d%s" % (i, extention))
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
	plt.savefig("results/highres/exerp0%d_model_76790_highres_nowhitespace.h5%s" % (i + 24, extention))
	print(i)
	# no whitespace image
	# savefig('foo.png', bbox_inches='tight')
	i += 1
