mod graphics;

use enum_ordinalize::Ordinalize;
use graphics::{Color, ColorMixer, ImageGrid};
use image::{GenericImage, Rgb, RgbImage, SubImage};
use ndarray::{s, Array2, Array3, Axis};
use rand::prelude::*;
use show_image::{
  create_window,
  event::{VirtualKeyCode, WindowEvent},
  WindowOptions,
};

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
  run()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
  let mut rng = thread_rng();

  let map = Map::gen(&mut rng, 100, 100);

  let people: Vec<_> = (0..100)
    .map(|_| Person {
      brain: Brain::gen(&mut rng, map.width(), map.height()),
      x: rng.gen_range(0..map.width()),
      y: rng.gen_range(0..map.height()),
    })
    .collect();

  let mut state = State {
    map,
    people,
    selected_person: 0,
  };

  let window = create_window(
    "Info Distribution",
    WindowOptions {
      size: Some([1920, 1080]),
      ..Default::default()
    },
  )?;
  window.set_image("image", state.draw())?;

  for event in window.event_channel()? {
    if let WindowEvent::KeyboardInput(event) = event {
      if !event.input.state.is_pressed() {
        continue;
      }
      match event.input.key_code {
        Some(VirtualKeyCode::Escape) => return Ok(()),
        Some(VirtualKeyCode::Left) => state.select_previous_person(),
        Some(VirtualKeyCode::Right) => state.select_next_person(),
        Some(VirtualKeyCode::Space) => state.update(),
        _ => continue,
      }
      window.set_image("image", state.draw())?;
    }
  }

  Ok(())
}

struct State {
  map: Map,
  people: Vec<Person>,
  selected_person: usize,
}

impl State {
  fn select_previous_person(&mut self) {
    self.selected_person = self
      .selected_person
      .checked_sub(1)
      .unwrap_or(self.people.len() - 1);
  }

  fn select_next_person(&mut self) {
    self.selected_person = (self.selected_person + 1) % self.people.len();
  }

  fn update(&mut self) {
    let mut rng = thread_rng();

    for person in self.people.iter_mut() {
      person.brain.0.map_inplace(|v| v.resdistribute(0.001));
    }

    for person in self.people.iter_mut() {
      let sense_range = 5;
      let min_x = person.x.saturating_sub(sense_range);
      let max_x = (person.x + sense_range).min(self.map.width());

      let min_y = person.y.saturating_sub(sense_range);
      let max_y = (person.y + sense_range).min(self.map.height());

      person
        .brain
        .0
        .slice_mut(s![min_x..max_x, min_y..max_y])
        .zip_mut_with(
          &self.map.0.slice(s![min_x..max_x, min_y..max_y]),
          |b, m| b.average_assign(&ResourceProbability::certain(*m)),
        );
    }

    let mut shuffled_mut: Vec<_> = self.people.iter_mut().collect();
    shuffled_mut.shuffle(&mut rng);
    let mut iter = shuffled_mut.into_iter();
    while iter.len() != 0 {
      let a = if let Some(a) = iter.next() {
        a
      } else {
        break;
      };

      let b = if let Some(b) = iter.next() {
        b
      } else {
        break;
      };

      let tmp = a.brain.clone();
      a.brain.0.zip_mut_with(&b.brain.0, |a, b| {
        a.average_assign(b);
      });
      b.brain.0.zip_mut_with(&tmp.0, |a, b| {
        a.average_assign(b);
      });
    }
  }

  fn draw(&self) -> RgbImage {
    let mut img = ImageGrid::new(self.map.width(), self.map.height(), 2, 2);

    {
      let mut buffer = img.grid_mut(0, 0);

      self.map.draw(&mut buffer);

      for (i, person) in self.people.iter().enumerate() {
        *buffer.get_pixel_mut(person.x as u32, person.y as u32) =
          if self.selected_person == i {
            Rgb([255, 255, 255])
          } else {
            Rgb([255, 255, 0])
          };
      }
    }

    {
      let mut buffer = img.grid_mut(0, 1);

      let mut votes = Array3::from_elem(
        (
          self.map.width(),
          self.map.height(),
          Resource::variant_count(),
        ),
        0,
      );

      for person in self.people.iter() {
        for ((x, y), v) in person.brain.0.indexed_iter() {
          *votes
            .get_mut((x, y, v.plurality().ordinal() as usize))
            .unwrap() += 1;
        }
      }

      let collective_view = votes.map_axis(Axis(2), |votes| {
        Resource::from_ordinal(
          votes.indexed_iter().max_by_key(|(_, v)| **v).unwrap().0 as i8,
        )
        .unwrap()
      });

      let error = Array2::from_shape_vec(
        self.map.0.raw_dim(),
        collective_view
          .iter()
          .zip(self.map.0.iter())
          .map(|(b, m)| b != m)
          .collect(),
      )
      .unwrap();

      for ((x, y), v) in error.indexed_iter() {
        let as_u8 = if *v { 255 } else { 0 };
        *buffer.get_pixel_mut(x as u32, y as u32) = Rgb([as_u8, as_u8, as_u8]);
      }
    }

    let selected_person = &self.people[self.selected_person];

    {
      let mut buffer = img.grid_mut(1, 0);

      selected_person.brain.draw(&mut buffer);
    }

    {
      let mut buffer = img.grid_mut(1, 1);

      let error = Array2::from_shape_vec(
        self.map.0.raw_dim(),
        selected_person
          .brain
          .0
          .map(|v| v.plurality())
          .iter()
          .zip(self.map.0.iter())
          .map(|(b, m)| b != m)
          .collect(),
      )
      .unwrap();

      for ((x, y), v) in error.indexed_iter() {
        let as_u8 = if *v { 255 } else { 0 };
        *buffer.get_pixel_mut(x as u32, y as u32) = Rgb([as_u8, as_u8, as_u8]);
      }
    }

    img.into_inner()
  }
}

struct Map(Array2<Resource>);

impl Map {
  fn gen<R: Rng>(rng: &mut R, width: usize, height: usize) -> Self {
    Self(Array2::from_shape_simple_fn((width, height), || {
      Resource::from_ordinal(rng.gen_range(0..Resource::variant_count() as i8))
        .unwrap()
    }))
  }

  fn width(&self) -> usize {
    self.0.shape()[0]
  }

  fn height(&self) -> usize {
    self.0.shape()[1]
  }

  fn draw(&self, img: &mut SubImage<&mut RgbImage>) {
    for ((x, y), v) in self.0.indexed_iter() {
      *img.get_pixel_mut(x as u32, y as u32) = v.color().into();
    }
  }
}

#[derive(Clone)]
struct Brain(Array2<ResourceProbability>);

impl Brain {
  fn gen<R: Rng>(rng: &mut R, width: usize, height: usize) -> Self {
    Self(Array2::from_shape_simple_fn((width, height), || {
      ResourceProbability::gen(rng)
    }))
  }

  fn width(&self) -> usize {
    self.0.shape()[0]
  }

  fn height(&self) -> usize {
    self.0.shape()[1]
  }

  fn draw(&self, img: &mut SubImage<&mut RgbImage>) {
    for ((x, y), v) in self.0.indexed_iter() {
      *img.get_pixel_mut(x as u32, y as u32) = Color::from(v).into();
    }
  }
}

struct Person {
  brain: Brain,
  x: usize,
  y: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Ordinalize)]
enum Resource {
  None,
  Food,
  Water,
  Stone,
}

impl Resource {
  fn color(&self) -> Color {
    match self {
      Resource::None => [0.0, 0.0, 0.0],
      Resource::Food => [0.0, 1.0, 0.0],
      Resource::Water => [0.0, 0.0, 1.0],
      Resource::Stone => [0.5, 0.5, 0.5],
    }
    .into()
  }
}

#[derive(Clone)]
struct ResourceProbability([f64; Resource::variant_count()]);

impl ResourceProbability {
  fn gen<R: Rng>(rng: &mut R) -> Self {
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

  fn certain(resource: Resource) -> Self {
    let mut inner = [0.0; Resource::variant_count()];
    inner[resource.ordinal() as usize] = 1.0;
    Self(inner)
  }

  fn plurality(&self) -> Resource {
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

  fn resdistribute(&mut self, percent: f64) {
    let len = self.0.len() as f64;
    for v in self.0.iter_mut() {
      *v -= *v * percent;
      *v += percent / len;
    }
  }

  fn average_assign(&mut self, other: &Self) {
    for (v, o) in self.0.iter_mut().zip(other.0.iter()) {
      *v = (*v + *o) / 2.0;
    }
  }
}

impl From<&ResourceProbability> for Color {
  fn from(rp: &ResourceProbability) -> Self {
    let mut mixer = ColorMixer::new();
    for (i, p) in rp.0.iter().enumerate() {
      mixer.mix_weighted(&Resource::from_ordinal(i as i8).unwrap().color(), *p);
    }
    mixer.into()
  }
}
