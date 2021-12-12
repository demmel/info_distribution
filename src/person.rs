use image::GenericImage;
use image::Rgb;
use image::RgbImage;
use image::SubImage;
use ndarray::Array2;
use rand::prelude::*;

use crate::graphics::Color;
use crate::resource::Resource;

#[derive(Clone)]
pub(crate) struct Brain {
  pub(crate) map: Array2<ResourceProbability>,
  pub(crate) home: (usize, usize),
  pub(crate) dest: (usize, usize),
}

impl Brain {
  pub(crate) fn gen<R: Rng>(rng: &mut R, width: usize, height: usize) -> Self {
    Self {
      map: Array2::from_shape_simple_fn((width, height), || {
        ResourceProbability::gen(rng)
      }),
      home: (rng.gen_range(0..width), rng.gen_range(0..height)),
      dest: (rng.gen_range(0..width), rng.gen_range(0..height)),
    }
  }

  pub(crate) fn draw(&self, img: &mut SubImage<&mut RgbImage>) {
    for ((x, y), v) in self.map.indexed_iter() {
      *img.get_pixel_mut(x as u32, y as u32) = Color::from(v).into();
    }

    *img.get_pixel_mut(self.home.0 as u32, self.home.1 as u32) =
      Rgb([255, 255, 0]);

    *img.get_pixel_mut(self.dest.0 as u32, self.dest.1 as u32) =
      Rgb([255, 0, 255]);
  }
}

pub(crate) struct Person {
  pub(crate) brain: Brain,
  pub(crate) x: usize,
  pub(crate) y: usize,
}

#[derive(Clone)]
pub(crate) struct ResourceProbability([f64; Resource::variant_count()]);

impl ResourceProbability {
  pub(crate) fn gen<R: Rng>(rng: &mut R) -> Self {
    let mut inner = [0.0; Resource::variant_count()];
    for v in inner.iter_mut() {
      *v = rng.gen();
    }
    let total: f64 = inner.iter().sum();
    for v in inner.iter_mut() {
      *v /= total;
    }
    Self(inner)
  }

  pub(crate) fn probable(resource: Resource, p: f64) -> Self {
    let even = 1.0 / Resource::variant_count() as f64;
    let subject = even + (1.0 - even) * p;
    let rest = (1.0 - subject) / (Resource::variant_count() - 1) as f64;
    let mut inner = [rest; Resource::variant_count()];
    inner[resource.ordinal() as usize] = subject;
    Self(inner)
  }

  pub(crate) fn plurality(&self) -> Resource {
    Resource::from_ordinal(
      self
        .0
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as i8,
    )
    .unwrap()
  }

  pub(crate) fn normalize(&mut self) {
    let total: f64 = self.0.iter().sum();
    for v in self.0.iter_mut() {
      *v /= total;
    }
  }

  pub(crate) fn resdistribute(&mut self, percent: f64) {
    let len = self.0.len() as f64;
    for v in self.0.iter_mut() {
      *v -= *v * percent;
      *v += percent / len;
    }
  }

  pub(crate) fn adjust_towards(&mut self, other: &Self, trust: f64) {
    for (v, o) in self.0.iter_mut().zip(other.0.iter()) {
      let bias = (0.5 - *v).abs() / 0.5;
      let other_bias = (0.5 - *o).abs() / 0.5;
      let trust_influence = (0.5 - trust).abs() / 0.5;
      let bias_weight = (1.0 - trust) * trust_influence
        + (bias * other_bias * 0.5 + 0.5) * (1.0 - trust_influence);
      *v = *v * bias_weight + *o * (1.0 - bias_weight);
    }
    self.normalize();
  }
}

impl From<&ResourceProbability> for Color {
  fn from(rp: &ResourceProbability) -> Self {
    let mut mixer = crate::graphics::ColorMixer::new();
    for (i, p) in rp.0.iter().enumerate() {
      mixer.mix_weighted(&Resource::from_ordinal(i as i8).unwrap().color(), *p);
    }
    mixer.into()
  }
}
