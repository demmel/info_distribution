use enum_ordinalize::Ordinalize;
use image::GenericImage;
use image::RgbImage;
use image::SubImage;
use ndarray::Array2;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::config::NUM_BIOMES;
use crate::resource::Resource;

pub(crate) struct Map {
  pub(crate) resources: Array2<Resource>,
  pub(crate) biomes: Biomes,
}

impl Map {
  pub(crate) fn gen<R: Rng>(rng: &mut R, width: usize, height: usize) -> Self {
    let biomes: Vec<_> = (0..NUM_BIOMES)
      .map(|_| {
        (
          rng.gen_range(0..width),
          rng.gen_range(0..height),
          Biome::gen(rng),
        )
      })
      .collect();

    let biomes = Biomes(biomes);
    Self {
      resources: Array2::from_shape_fn((width, height), |(x, y)| {
        biomes.get_biome(x, y).gen_resource(rng)
      }),
      biomes,
    }
  }

  pub(crate) fn width(&self) -> usize {
    self.resources.shape()[0]
  }

  pub(crate) fn height(&self) -> usize {
    self.resources.shape()[1]
  }

  pub(crate) fn draw(&self, img: &mut SubImage<&mut RgbImage>) {
    for ((x, y), v) in self.resources.indexed_iter() {
      img.put_pixel(x as u32, y as u32, v.color().into());
    }
  }
}

pub(crate) struct Biomes(Vec<(usize, usize, Biome)>);

impl Biomes {
  pub(crate) fn get_biome(&self, x: usize, y: usize) -> &Biome {
    self
      .0
      .iter()
      .map(|(ox, oy, b)| {
        (
          ((x as f64 - *ox as f64).powi(2) + (y as f64 - *oy as f64).powi(2))
            .sqrt(),
          b,
        )
      })
      .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
      .unwrap()
      .1
  }
}

#[derive(Ordinalize)]
pub(crate) enum Biome {
  Plains,
  Lake,
  Mountain,
  Graveyard,
}

impl Biome {
  pub(crate) fn gen<R: Rng>(rng: &mut R) -> Self {
    Biome::from_ordinal(rng.gen_range(0..Biome::variant_count() as i8)).unwrap()
  }

  pub(crate) fn gen_resource<R: Rng>(&self, rng: &mut R) -> Resource {
    let weights = match self {
      Biome::Plains => [0.2, 0.75, 0.0, 0.05, 0.0],
      Biome::Lake => [0.0, 0.0, 1.0, 0.0, 0.0],
      Biome::Mountain => [0.05, 0.05, 0.05, 0.85, 0.0],
      Biome::Graveyard => [0.9, 0.0, 0.0, 0.05, 0.05],
    };
    Resource::from_ordinal(
      WeightedIndex::new(weights).unwrap().sample(rng) as i8
    )
    .unwrap()
  }
}
