use image::GenericImage;
use image::RgbImage;
use image::SubImage;
use ndarray::Array2;
use rand::prelude::*;

use crate::resource::Resource;

pub(crate) struct Map(pub(crate) Array2<Resource>);

impl Map {
  pub(crate) fn gen<R: Rng>(rng: &mut R, width: usize, height: usize) -> Self {
    Self(Array2::from_shape_simple_fn((width, height), || {
      Resource::from_ordinal(rng.gen_range(0..Resource::variant_count() as i8))
        .unwrap()
    }))
  }

  pub(crate) fn width(&self) -> usize {
    self.0.shape()[0]
  }

  pub(crate) fn height(&self) -> usize {
    self.0.shape()[1]
  }

  pub(crate) fn draw(&self, img: &mut SubImage<&mut RgbImage>) {
    for ((x, y), v) in self.0.indexed_iter() {
      *img.get_pixel_mut(x as u32, y as u32) = v.color().into();
    }
  }
}
