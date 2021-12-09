use std::ops::Mul;

use image::{GenericImage, ImageBuffer, Rgb, RgbImage, SubImage};

pub struct ImageGrid {
  img: RgbImage,
  width: usize,
  height: usize,
}

impl ImageGrid {
  pub fn new(width: usize, height: usize, c: usize, r: usize) -> Self {
    let img: RgbImage = ImageBuffer::new(
      (c * width + (c - 1)) as u32,
      (r * height + (r - 1)) as u32,
    );
    Self { img, width, height }
  }

  pub fn into_inner(self) -> RgbImage {
    self.img
  }

  pub fn grid_mut(&mut self, c: usize, r: usize) -> SubImage<&mut RgbImage> {
    self.img.sub_image(
      (c * (self.width + 1)) as u32,
      (r * (self.height + 1)) as u32,
      self.width as u32,
      self.height as u32,
    )
  }
}

pub struct Color([f64; 3]);

impl Mul<f64> for Color {
  type Output = Color;

  fn mul(mut self, rhs: f64) -> Self::Output {
    for v in self.0.iter_mut() {
      *v *= rhs;
    }
    self
  }
}

impl From<[f64; 3]> for Color {
  fn from(inner: [f64; 3]) -> Self {
    Color(inner)
  }
}

impl From<Color> for Rgb<u8> {
  fn from(color: Color) -> Self {
    [
      (color.0[0] * 255.0) as u8,
      (color.0[1] * 255.0) as u8,
      (color.0[2] * 255.0) as u8,
    ]
    .into()
  }
}

pub struct ColorMixer {
  mixer: [f64; 3],
  weight: f64,
}

impl ColorMixer {
  pub fn new() -> Self {
    Self {
      mixer: [0.0; 3],
      weight: 0.0,
    }
  }

  fn mix(&mut self, color: &Color) {
    for (v, c) in self.mixer.iter_mut().zip(color.0) {
      *v += c;
    }
    self.weight += 1.0;
  }

  pub fn mix_weighted(&mut self, color: &Color, weight: f64) {
    for (v, c) in self.mixer.iter_mut().zip(color.0) {
      *v += weight * c;
    }
    self.weight += weight
  }
}

impl Default for ColorMixer {
  fn default() -> Self {
    Self::new()
  }
}

impl From<ColorMixer> for Color {
  fn from(mixer: ColorMixer) -> Self {
    let ColorMixer { mut mixer, weight } = mixer;
    for v in mixer.iter_mut() {
      *v /= weight
    }
    Color(mixer)
  }
}
