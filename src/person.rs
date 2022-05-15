use image::GenericImage;
use image::RgbImage;
use image::SubImage;
use ndarray::array;
use ndarray::Array2;
use rand::prelude::*;

use crate::config::MAX_HUNGER;
use crate::config::MAX_THIRST;
use crate::graphics::Color;
use crate::ndarray_pad::ArrayPaddingExt;
use crate::ndarray_pad::ArrayPaddingKind;
use crate::resource::Resource;

pub(crate) struct Person {
  pub(crate) brain: Brain,
  pub(crate) needs: Needs,
  pub(crate) x: usize,
  pub(crate) y: usize,
}

impl Person {
  pub(crate) fn favorability_map(&self) -> Array2<f64> {
    let hunger_percent = self.needs.hunger as f64 / MAX_HUNGER as f64;
    let thirst_percent = self.needs.thirst as f64 / MAX_THIRST as f64;

    let mut favorability = Array2::from_shape_vec(
      self.brain.map.raw_dim(),
      self
        .brain
        .map
        .map(|b| {
          b.get(Resource::Food) * hunger_percent
            + b.get(Resource::Water) * thirst_percent
            - b.get(Resource::Ghost) * 1.0
        })
        .pad((2, 2), ArrayPaddingKind::Constant(0.0))
        .windows((5, 5))
        .into_iter()
        .map(|w| {
          (array![
            [1.0f64, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 100.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
          ] * w)
            .fold(0.0, |acc, cur| acc + cur)
        })
        .collect(),
    )
    .unwrap();

    for ((x, y), f) in favorability.indexed_iter_mut() {
      let dist = ((self.x as f64 - x as f64).powi(2)
        + (self.y as f64 - y as f64).powi(2))
      .sqrt();
      *f *= 0.9 / (dist + 1.0) + 0.1;
    }

    favorability
  }
}

pub(crate) struct Needs {
  pub(crate) hunger: u16,
  pub(crate) thirst: u16,
}

impl Needs {
  pub(crate) fn met(&self) -> bool {
    self.hunger < MAX_HUNGER && self.thirst < MAX_THIRST
  }
}

#[derive(Clone)]
pub(crate) struct Brain {
  pub(crate) map: Array2<ResourceProbability>,
}

impl Brain {
  pub(crate) fn gen<R: Rng>(rng: &mut R, width: usize, height: usize) -> Self {
    Self {
      map: Array2::from_shape_simple_fn((width, height), || {
        ResourceProbability::gen(rng)
      }),
    }
  }

  pub(crate) fn draw(&self, img: &mut SubImage<&mut RgbImage>) {
    for ((x, y), v) in self.map.indexed_iter() {
      img.put_pixel(x as u32, y as u32, Color::from(v).into());
    }
  }
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

  pub(crate) fn get(&self, resource: Resource) -> f64 {
    self.0[resource.ordinal() as usize]
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
