import numpy as np

from hcbf.corruptions import brightness, gaussian_noise, horizontal_motion_blur


def test_gaussian_noise_is_repeatable_and_realization_specific():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    a = gaussian_noise(image, sigma=25, sample_id="sample", severity="moderate", realization=0)
    b = gaussian_noise(image, sigma=25, sample_id="sample", severity="moderate", realization=0)
    c = gaussian_noise(image, sigma=25, sample_id="sample", severity="moderate", realization=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_brightness_and_blur_preserve_shape_and_dtype():
    image = np.arange(9 * 9, dtype=np.uint8).reshape(9, 9)
    assert brightness(image, factor=0.5).shape == image.shape
    blurred = horizontal_motion_blur(image, kernel_size=5)
    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype
