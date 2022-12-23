import numpy as np
import tensorflow as tf
import cv2 as cv
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

def SE(a,b):
    return tf.reduce_sum(tf.square(a-b))
  
def scale_invariant_mse(prediction, base):
    n = float(base.size)
    log_diff = tf.log(prediction+1e-4) - tf.log(base+1e-4)
    return tf.sqrt(tf.reduce_sum(tf.square(log_diff)) / n - tf.square(tf.reduce_sum(log_diff)) / tf.square(n))

def high_pass_filter(img,mask):
    image_fft=tf.signal.fft2d(tf.cast(img,tf.complex64))
    image_shift=tf.signal.fftshift(image_fft)
    image_fft_mask=image_shift*mask
    image_ifft_mask=tf.signal.ifft2d(image_fft_mask)
    return image_ifft_mask

def contrast(image):
    return image/2 + 63

def samon_vgg(input_shape):
    model = VGG16(include_top=False,weights="imagenet",input_shape = input_shape)
    layers = model.layers[1:19]
    layer_outputs = [layer.output for layer in layers]
    return Model(inputs=model.input, outputs=layer_outputs)

def ZNCC(pre, base):
    mean = tf.reduce_mean(pre)
    mean2 = tf.reduce_mean(base)
    im1 = pre - mean
    im2 = base - mean2
    num = tf.reduce_sum(im1 * im2)
    den = tf.sqrt(tf.reduce_sum(im1 * im1)) * tf.sqrt(tf.reduce_sum(im2 * im2))
    return tf.abs(num / den)

#眼鏡用
image=cv.imread("SS1.png")#写真を入力してください
#裸眼用
image2=cv.imread("SS5.png")
image=contrast(image)
image_R = image[:,:,0]
image_G = image[:,:,1]
image_B = image[:,:,2]

image2=contrast(image2)
image2_R = image2[:,:,0]
image2_G = image2[:,:,1]
image2_B = image2[:,:,2]

psf=np.load("psf_05.npy")
psf=cv.resize(psf,(35,35)) #実際使用環境を想定
psf_fix=np.load("psf_seishi.npy")
config = tf.ConfigProto(
    device_count={"GPU":0}, # GPUの数0に
    log_device_placement=True,
    allow_soft_placement=True	
)

#VGG16
input_shape = (150, 150, 3)
activation_model = samon_vgg(input_shape)

array_i_R=np.reshape(image_R,(1,image.shape[0],image.shape[1],1)).astype(np.float)
array_i_G=np.reshape(image_G, (1, image.shape[0],image.shape[1],1)).astype(np.float)
array_i_B=np.reshape(image_B, (1, image.shape[0],image.shape[1],1)).astype(np.float)

array_i2_R=np.reshape(image2_R,(1,image2.shape[0],image2.shape[1],1)).astype(np.float)
array_i2_G=np.reshape(image2_G,(1,image2.shape[0],image2.shape[1],1)).astype(np.float)
array_i2_B=np.reshape(image2_B,(1,image2.shape[0],image2.shape[1],1)).astype(np.float)

point_psf=np.reshape(psf_fix,(psf_fix.shape[0],psf_fix.shape[1],1,1)).astype(np.float)

image_in_R = tf.placeholder(dtype=tf.float32,shape=[1,None,None,1])
image_in_G = tf.placeholder(dtype=tf.float32,shape=[1,None,None,1])
image_in_B = tf.placeholder(dtype=tf.float32,shape=[1,None,None,1])
image_in_2_R = tf.placeholder(dtype=tf.float32,shape=[1,None,None,1])
image_in_2_G = tf.placeholder(dtype=tf.float32,shape=[1,None,None,1])
image_in_2_B = tf.placeholder(dtype=tf.float32,shape=[1,None,None,1])
point_psf_tensor=tf.placeholder(dtype=tf.float32,shape=[None,None,1,1])

display_num = 4
displays_R = []
displays_G = []
displays_B = []
glasses = []
maked_psfs = []
xTy_glasses_R = []
xTy_glasses_G = []
xTy_glasses_B = []

xTy_nakeds_R = []
xTy_nakeds_G = []
xTy_nakeds_B = []

weights = []
omomi = 100000000
for num in range(display_num):
    displays_R.append(tf.Variable(tf.random_uniform(shape=[1,image.shape[0],image.shape[1],1],minval=0.0,maxval=1.0,dtype=tf.float32))*250)
    displays_G.append(tf.Variable(tf.random_uniform(shape=[1,image.shape[0],image.shape[1],1],minval=0.0,maxval=1.0,dtype=tf.float32))*250)
    displays_B.append(tf.Variable(tf.random_uniform(shape=[1,image.shape[0],image.shape[1],1],minval=0.0,maxval=1.0,dtype=tf.float32))*250)
    displays_R[num] = tf.math.minimum(tf.math.maximum(displays_R[num],0),255)
    displays_G[num] = tf.math.minimum(tf.math.maximum(displays_G[num],0),255)
    displays_B[num] = tf.math.minimum(tf.math.maximum(displays_B[num],0),255)
    
  

    glasses.append(tf.Variable(tf.random_uniform([psf.shape[0],psf.shape[1]],minval=0.0,maxval=1.0, dtype=tf.float32)))
    glasses[num] = tf.math.minimum(tf.math.maximum(glasses[num],0),2.0)

    temp = psf*glasses[num]
    maked_psfs.append(tf.reshape(temp,(psf.shape[0],psf.shape[1],1,1)))

    xTy_glasses_R.append(tf.nn.conv2d(displays_R[num], maked_psfs[num], strides=[1,1,1,1],padding='SAME'))
    xTy_glasses_G.append(tf.nn.conv2d(displays_G[num], maked_psfs[num], strides=[1,1,1,1],padding='SAME'))
    xTy_glasses_B.append(tf.nn.conv2d(displays_B[num], maked_psfs[num], strides=[1,1,1,1],padding='SAME'))
    xTy_nakeds_R.append(tf.nn.conv2d(displays_R[num], point_psf_tensor, strides=[1,1,1,1],padding='SAME'))
    xTy_nakeds_G.append(tf.nn.conv2d(displays_G[num], point_psf_tensor, strides=[1,1,1,1],padding='SAME'))
    xTy_nakeds_B.append(tf.nn.conv2d(displays_B[num], point_psf_tensor, strides=[1,1,1,1],padding='SAME'))

    weights.append(tf.math.maximum(np.sum(psf)*1.0-tf.reduce_sum(maked_psfs[num]),0)*omomi)

xTy_glass_R = 0
xTy_glass_G = 0
xTy_glass_B = 0

xTy_naked_R = 0
xTy_naked_G = 0
xTy_naked_B = 0

weight = 0
for num in range(display_num):
    xTy_glass_R += xTy_glasses_R[num]
    xTy_glass_G += xTy_glasses_G[num]
    xTy_glass_B += xTy_glasses_B[num]
    xTy_naked_R += xTy_nakeds_R[num]
    xTy_naked_G += xTy_nakeds_G[num]
    xTy_naked_B += xTy_nakeds_B[num]
    weight += weights[num]
xTy_glass_R /= display_num
xTy_glass_G /= display_num
xTy_glass_B /= display_num
xTy_naked_R /= display_num
xTy_naked_G /= display_num
xTy_naked_B /= display_num
weight /= display_num


#ハイパスフィルタ
size=image_R.shape
mask=np.zeros(size)
length=size[0]
centery=size[0]/2
R=10
for x in range(0,length):
    for y in range(0,length):
        if(x-centery)**2 + (y-centery)**2 >R**2:
            mask[x,y]=1
reshaped_xTy_glass_R=tf.reshape(xTy_glass_R,(size[0],size[1]))
reshaped_xTy_glass_G=tf.reshape(xTy_glass_G,(size[0],size[1]))
reshaped_xTy_glass_B=tf.reshape(xTy_glass_B,(size[0],size[1]))
reshaped_xTy_naked_R=tf.reshape(xTy_naked_R,(size[0],size[1]))
reshaped_xTy_naked_G=tf.reshape(xTy_naked_G,(size[0],size[1]))
reshaped_xTy_naked_B=tf.reshape(xTy_naked_B,(size[0],size[1]))


target_glass_fft_R=image_R
target_glass_fft_G=image_G
target_glass_fft_B=image_B

target_naked_fft_R=image2_R
target_naked_fft_G=image2_G
target_naked_fft_B=image2_B


glass_hpf_R=high_pass_filter(reshaped_xTy_glass_R, mask)
glass_hpf_G=high_pass_filter(reshaped_xTy_glass_G, mask)
glass_hpf_B=high_pass_filter(reshaped_xTy_glass_B, mask)
naked_hpf_R=high_pass_filter(reshaped_xTy_naked_R, mask)
naked_hpf_G=high_pass_filter(reshaped_xTy_naked_G, mask)
naked_hpf_B=high_pass_filter(reshaped_xTy_naked_B, mask)

target_glass_hpf_R=high_pass_filter(target_glass_fft_R,mask)
target_glass_hpf_G=high_pass_filter(target_glass_fft_G,mask)
target_glass_hpf_B=high_pass_filter(target_glass_fft_B,mask)
target_naked_hpf_R=high_pass_filter(target_naked_fft_R,mask)
target_naked_hpf_G=high_pass_filter(target_naked_fft_G,mask)
target_naked_hpf_B=high_pass_filter(target_naked_fft_B,mask)

ls_hpf_R = tf.cast(SE(target_glass_hpf_R,glass_hpf_R),tf.float32)*10+tf.cast(SE(target_naked_hpf_R,naked_hpf_R),tf.float32)*10
ls_hpf_G = tf.cast(SE(target_glass_hpf_G,glass_hpf_G),tf.float32)*10+tf.cast(SE(target_naked_hpf_G,naked_hpf_G),tf.float32)*10
ls_hpf_B = tf.cast(SE(target_glass_hpf_B,glass_hpf_B),tf.float32)*10+tf.cast(SE(target_naked_hpf_B,naked_hpf_B),tf.float32)*10

loss_of_hpf=ls_hpf_R + ls_hpf_G + ls_hpf_B


#VGG16　提示画像と眼鏡側目標画像の意味距離を最大化する
#眼鏡側目標画像
vgg_target_R = preprocess_input(np.concatenate([array_i_R, array_i_R, array_i_R], -1))
vgg_target_G = preprocess_input(np.concatenate([array_i_G, array_i_G, array_i_G], -1))
vgg_target_B = preprocess_input(np.concatenate([array_i_B, array_i_B, array_i_B], -1))
act_target_R = activation_model.predict(vgg_target_R)
act_target_G = activation_model.predict(vgg_target_G)
act_target_B = activation_model.predict(vgg_target_B)
#提示画像
preprocess_images_R = []
preprocess_images_G = []
preprocess_images_B = []
acts_R = []
acts_G = []
acts_B = []
N = 75*75
sums_simse = 0
sums_zncc = 0
for num in range(display_num):
    preprocess_images_R.append(preprocess_input(tf.concat([displays_R[num],displays_R[num],displays_R[num]], -1)))
    preprocess_images_G.append(preprocess_input(tf.concat([displays_G[num],displays_G[num],displays_G[num]], -1)))
    preprocess_images_B.append(preprocess_input(tf.concat([displays_B[num],displays_B[num],displays_B[num]], -1)))

    acts_R.append(activation_model(preprocess_images_R[num]))
    acts_G.append(activation_model(preprocess_images_G[num]))
    acts_B.append(activation_model(preprocess_images_B[num]))

    for choose in range(3,6):
        sums_simse += scale_invariant_mse(acts_R[num][choose], act_target_R[choose])
        sums_simse += scale_invariant_mse(acts_G[num][choose], act_target_G[choose])
        sums_simse += scale_invariant_mse(acts_B[num][choose], act_target_B[choose])

        sums_zncc += ZNCC(acts_R[num][choose], act_target_R[choose])
        sums_zncc += ZNCC(acts_G[num][choose], act_target_G[choose])
        sums_zncc += ZNCC(acts_B[num][choose], act_target_B[choose])
#SIMSE
sums_simse = sums_simse*1e7*5/3
#ZNCC
sums_zncc = sums_zncc*1e10/4
sums = 0
sums += sums_simse
sums -= sums_zncc

loss_R=tf.reduce_sum(tf.square(image_in_R-xTy_glass_R))*2 + tf.reduce_sum(tf.square(image_in_2_R-xTy_naked_R))
loss_G=tf.reduce_sum(tf.square(image_in_R-xTy_glass_G))*2 + tf.reduce_sum(tf.square(image_in_2_G-xTy_naked_G))
loss_B=tf.reduce_sum(tf.square(image_in_R-xTy_glass_B))*2 + tf.reduce_sum(tf.square(image_in_2_B-xTy_naked_B))

loss=loss_R + loss_G + loss_B + weight + loss_of_hpf * (display_num / 2) - sums*3
lr=tf.placeholder(tf.float64,shape=[])
opt=tf.train.AdamOptimizer(learning_rate=lr)
rec=opt.minimize(loss)
#g=s*m*k
print("start")
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

lr_base=0.001
lr_ratio=10
ls=sess.run(loss,feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
print(ls)
for i in range(1500):
    nm = sess.run([rec],feed_dict={lr:lr_base*lr_ratio,image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
    if(i%500==0):
        lr_ratio/=1.0
        ls=sess.run(loss,feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
        print(ls)
        #sums=sess.run([sums_simse, sums_zncc],feed_dict={image_in:array_i, image_in_2:array_i2, point_psf_tensor:point_psf})
        #print(ls,sums)
glass_image_R, naked_image_R = sess.run([xTy_glass_R, xTy_naked_R], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
glass_image_G, naked_image_G = sess.run([xTy_glass_G, xTy_naked_G], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
glass_image_B, naked_image_B = sess.run([xTy_glass_B, xTy_naked_B], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})

glass_masks = []
psf_results = []
display_results_R = []
display_results_G = []
display_results_B = []
for num in range(display_num):
    temp_result = sess.run(glasses[num], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
    glass_masks.append(temp_result)

    temp_result = sess.run(maked_psfs[num], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
    psf_results.append(temp_result)

    temp_result_R = sess.run(displays_R[num], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
    temp_result_G = sess.run(displays_G[num], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})
    temp_result_B = sess.run(displays_B[num], feed_dict={image_in_R:array_i_R,image_in_G:array_i_G,image_in_B:array_i_B, image_in_2_R:array_i2_R,image_in_2_G:array_i2_G,image_in_2_B:array_i2_B, point_psf_tensor:point_psf})

    display_results_R.append(temp_result_R)
    display_results_G.append(temp_result_G)
    display_results_B.append(temp_result_B)

print("write start")
result_dir="color_SIMSE_ZNCC_contrast_zero_Separate/"
if(not os.path.exists(result_dir)):
    os.mkdir(result_dir)
result_dir2=result_dir+"parameters/"
if(not os.path.exists(result_dir2)):
    os.mkdir(result_dir2)
for num in range(display_num):
    np.save(result_dir2+"psf"+str(num+1)+".npy",psf_results[num])
    np.save(result_dir2+"glass"+str(num+1)+".npy",glass_masks[num])
print("write end")

glass_ob_R=np.reshape(glass_image_R,(image.shape[0],image.shape[1]))
glass_ob_G=np.reshape(glass_image_G,(image.shape[0],image.shape[1]))
glass_ob_B=np.reshape(glass_image_B,(image.shape[0],image.shape[1]))
glass_ob = np.zeros(image.shape)
glass_ob[:,:,0] = glass_ob_R
glass_ob[:,:,1] = glass_ob_G
glass_ob[:,:,2] = glass_ob_B
cv.imwrite(result_dir+"glass_ob.jpg",glass_ob)

naked_ob_R=np.reshape(naked_image_R,(image2.shape[0],image2.shape[1]))
naked_ob_G=np.reshape(naked_image_G,(image2.shape[0],image2.shape[1]))
naked_ob_B=np.reshape(naked_image_B,(image2.shape[0],image2.shape[1]))
naked_ob = np.zeros(image.shape)
naked_ob[:,:,0] = naked_ob_R
naked_ob[:,:,1] = naked_ob_G
naked_ob[:,:,2] = naked_ob_B

print(cv.imwrite(result_dir+"naked_ob.jpg",naked_ob))

for num in range(display_num):
    temp_psf = np.reshape(psf_results[num], (psf.shape[0],psf.shape[1]))
    temp_psf=temp_psf*255/np.max(temp_psf)
    cv.imwrite(result_dir+"psf_"+str(num+1)+".jpg",temp_psf)
    temp_display_R = np.reshape(display_results_R[num],(image.shape[0],image.shape[1]))
    temp_display_G = np.reshape(display_results_G[num],(image.shape[0],image.shape[1]))
    temp_display_B = np.reshape(display_results_B[num],(image.shape[0],image.shape[1]))
    temp_display = np.zeros(image.shape)
    temp_display[:,:,0] = temp_display_R
    temp_display[:,:,1] = temp_display_G
    temp_display[:,:,2] = temp_display_B
    cv.imwrite(result_dir+"display_"+str(num+1)+".png",temp_display)
