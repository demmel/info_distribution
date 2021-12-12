mod graphics;
mod map;
mod person;
mod resource;

use image::{GenericImage, Rgb, RgbImage};
use ndarray::{s, Array2, Array3, Axis, Zip};
use rand::prelude::*;
use show_image::{
  create_window,
  event::{VirtualKeyCode, WindowEvent},
  WindowOptions,
};

use crate::graphics::ImageGrid;
use crate::map::Map;
use crate::person::{Brain, Person, ResourceProbability};
use crate::resource::Resource;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
  run()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
  let mut rng = thread_rng();

  let mut state = State::gen(&mut rng);

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
  fn gen<R: Rng>(rng: &mut R) -> Self {
    let map = Map::gen(rng, 200, 200);

    let people: Vec<_> = (0..10)
      .map(|_| Person {
        brain: Brain::gen(rng, map.width(), map.height()),
        x: rng.gen_range(0..map.width()),
        y: rng.gen_range(0..map.height()),
      })
      .collect();

    Self {
      map,
      people,
      selected_person: 0,
    }
  }

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

    // Memory degradation
    for person in self.people.iter_mut() {
      person.brain.map.map_inplace(|v| v.resdistribute(0.001));
    }

    // Perception
    for person in self.people.iter_mut() {
      let sense_range = 10;

      let min_x = person.x.saturating_sub(sense_range);
      let max_x = (person.x + sense_range).min(self.map.width());

      let min_y = person.y.saturating_sub(sense_range);
      let max_y = (person.y + sense_range).min(self.map.height());

      Zip::indexed(person.brain.map.slice_mut(s![min_x..max_x, min_y..max_y]))
        .and(self.map.0.slice(s![min_x..max_x, min_y..max_y]))
        .for_each(|(x, y), b, m| {
          let x = min_x + x;
          let y = min_y + y;

          let dist = ((person.x as f64 - x as f64).powi(2)
            + (person.y as f64 - y as f64).powi(2))
          .sqrt();

          let certainty =
            (sense_range as f64 - dist).max(0.0) / sense_range as f64;

          b.adjust_towards(
            &ResourceProbability::probable(*m, certainty),
            certainty * certainty,
          );
        });
    }

    // Move
    for person in self.people.iter_mut() {
      let dx = person.brain.dest.0 as isize - person.x as isize;
      let dy = person.brain.dest.1 as isize - person.y as isize;

      if dx == 0 && dy == 0 {
        if person.x == person.brain.home.0 && person.y == person.brain.home.1 {
          person.brain.dest = (
            rng.gen_range(0..self.map.width()),
            rng.gen_range(0..self.map.height()),
          );
        } else {
          person.brain.dest = person.brain.home;
        }
      } else if dx.abs() > dy.abs() {
        person.x = (person.x as isize + dx.signum()) as usize;
      } else {
        person.y = (person.y as isize + dy.signum()) as usize;
      }
    }

    // Communication
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

      let (a_i, a_share): (Vec<_>, Vec<_>) = a
        .brain
        .map
        .indexed_iter()
        .choose_multiple(&mut rng, 100)
        .into_iter()
        .unzip();
      let a_share: Vec<_> =
        a_i.into_iter().zip(a_share.into_iter().cloned()).collect();
      let b_share = b.brain.map.indexed_iter().choose_multiple(&mut rng, 100);

      for (i, share) in b_share {
        a.brain.map[i].adjust_towards(share, 0.5);
      }
      for (i, share) in a_share {
        b.brain.map[i].adjust_towards(&share, 0.5);
      }
    }
  }

  fn draw(&self) -> RgbImage {
    let mut img = ImageGrid::new(self.map.width(), self.map.height(), 3, 2);
    let selected_person = &self.people[self.selected_person];

    {
      let mut buffer = img.grid_mut(0, 0);

      self.map.draw(&mut buffer);
    }

    {
      let mut buffer = img.grid_mut(0, 1);

      for (i, person) in self.people.iter().enumerate() {
        *buffer.get_pixel_mut(person.x as u32, person.y as u32) =
          if self.selected_person == i {
            Rgb([255, 255, 255])
          } else {
            Rgb([255, 0, 0])
          };
      }
    }

    {
      let mut buffer = img.grid_mut(1, 0);

      selected_person.brain.draw(&mut buffer);
    }

    {
      let mut buffer = img.grid_mut(2, 0);

      let error = Array2::from_shape_vec(
        self.map.0.raw_dim(),
        selected_person
          .brain
          .map
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

    {
      let mut buffer = img.grid_mut(2, 1);

      let mut votes = Array3::from_elem(
        (
          self.map.width(),
          self.map.height(),
          Resource::variant_count(),
        ),
        0,
      );

      for person in self.people.iter() {
        for ((x, y), v) in person.brain.map.indexed_iter() {
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

    img.into_inner()
  }
}
