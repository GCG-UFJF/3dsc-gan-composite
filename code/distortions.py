import tensorflow as tf
from fjcommon import tf_helpers


class Distortions(object):
    def __init__(self, x, x_out, distorction_function, axis, weights):
        assert tf.float32.is_compatible_with(x.dtype) and tf.float32.is_compatible_with(x_out.dtype)

        squared_error = tf.square(x_out - x)
        mse_per_image = tf.reduce_mean(squared_error, axis=[1, 2, 3, 4])
        psnr_per_image = -10 * tf_helpers.log10(mse_per_image)

        self.mse = tf.reduce_mean(mse_per_image)
        self.psnr = tf.reduce_mean(psnr_per_image)

        self.losses = distorction_function
        self.axis_order = axis
        self.weights = weights
        print(self.losses, self.axis_order, self.weights)

        self.d_loss = self.get_distortion(x, x_out)

    def get_distortion(self, x, x_out):
        # (B, 1, d, h, w)
        axis_aux = [0, 1, 2, 3, 4]

        loss = x_out - x

        for selected_loss, selected_axis, selected_weight in zip(self.losses, self.axis_order, self.weights):
            selected_axis   = int(selected_axis)
            selected_weight = float(selected_weight)

            if selected_loss == 'l1':
                loss = tf.reduce_sum(tf.abs(loss), axis=[axis_aux.index(selected_axis)])

            elif selected_loss == 'l2':
                loss = tf.reduce_sum(tf.square(loss), axis=[axis_aux.index(selected_axis)])

            elif selected_loss == 'mean':
                loss = tf.reduce_sum(loss, axis=[axis_aux.index(selected_axis)])

            elif selected_loss == 'huber':
                error = tf.reduce_sum(loss, axis=[axis_aux.index(selected_axis)])

                clip_delta = 1.0
                cond = tf.abs(error) < clip_delta

                squared_loss = 0.5 * tf.square(error)
                linear_loss  = clip_delta * (tf.abs(error) - 0.5 * clip_delta)

                loss = tf.where(cond, squared_loss, linear_loss)

            else: # selected_loss == 'logcosh'
                loss = tf.reduce_mean(loss + tf.log(tf.exp(-2. * loss) + 1.) - tf.log(2.), axis=[axis_aux.index(selected_axis)])

            loss *= selected_weight
            axis_aux.remove(selected_axis)

        loss = tf.reduce_mean(loss)
        return loss