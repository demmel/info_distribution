mod config;
mod graphics;
mod map;
mod ndarray_pad;
mod person;
mod resource;

use std::sync::mpsc::TryRecvError;

use config::{
  HUNGER_PER_FOOD, MAP_HEIGHT, MAP_WIDTH, NUM_PEROPLE, THIRST_PER_WATER,
};
use image::{GenericImage, Rgb, RgbImage};
use ndarray::{s, Array2, Array3, Axis, Zip};
use ndarray_pad::{ArrayPaddingExt, ArrayPaddingKind};
use person::Needs;
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
  let mut running = false;

  let window = create_window(
    "Info Distribution",
    WindowOptions {
      size: Some([1920, 1080]),
      ..Default::default()
    },
  )?;
  window.set_image("image", state.draw())?;

  let window_events = window.event_channel()?;
  loop {
    match window_events.try_recv() {
      Ok(WindowEvent::KeyboardInput(event)) => {
        if !event.input.state.is_pressed() {
          continue;
        }
        match event.input.key_code {
          Some(VirtualKeyCode::Escape) => return Ok(()),
          Some(VirtualKeyCode::Left) => state.select_previous_person(),
          Some(VirtualKeyCode::Right) => state.select_next_person(),
          Some(VirtualKeyCode::Space) if !running => state.update(),
          Some(VirtualKeyCode::S) => running = !running,
          _ => continue,
        }
        window.set_image("image", state.draw())?;
      }
      Err(TryRecvError::Empty) if running => {
        state.update();
        window.set_image("image", state.draw())?;
      }
      Err(TryRecvError::Disconnected) => return Ok(()),
      _ => continue,
    }
  }
}

struct State {
  map: Map,
  people: Vec<Person>,
  selected_person: usize,
}

impl State {
  fn gen<R: Rng>(rng: &mut R) -> Self {
    let map = Map::gen(rng, MAP_WIDTH, MAP_HEIGHT);

    let people: Vec<_> = (0..NUM_PEROPLE)
      .map(|_| Person {
        brain: Brain::gen(rng, map.width(), map.height()),
        x: rng.gen_range(0..map.width()),
        y: rng.gen_range(0..map.height()),
        needs: Needs {
          hunger: 0,
          thirst: 0,
        },
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

    // Map mutation
    {
      let paddded = self.map.resources.pad((1, 1), ArrayPaddingKind::Clamp);
      let windows: Vec<_> = paddded.windows((3, 3)).into_iter().collect();
      let resources = &mut self.map.resources;
      let biomes = &self.map.biomes;
      for (((x, y), r), w) in resources.indexed_iter_mut().zip(windows) {
        *r = match rng.gen_range(0..1_000_000) {
          0..=999_899 => w[(1, 1)],
          999_900..=999_989 => *w.iter().choose(&mut rng).unwrap(),
          999_990..=999_998 => biomes.get_biome(x, y).gen_resource(&mut rng),
          999_999 => *Resource::variants().choose(&mut rng).unwrap(),
          _ => unreachable!(),
        };
      }
    }

    // Needs increase
    for person in self.people.iter_mut() {
      person.needs.hunger += 1;
      person.needs.thirst += 1;
    }
    self.people.retain(|p| p.needs.met());

    // Memory degradation
    for person in self.people.iter_mut() {
      person.brain.map.map_inplace(|v| v.resdistribute(0.0001));
    }

    // Perception
    for person in self.people.iter_mut() {
      let sense_range = 10;

      let min_x = person.x.saturating_sub(sense_range);
      let max_x = (person.x + sense_range).min(self.map.width());

      let min_y = person.y.saturating_sub(sense_range);
      let max_y = (person.y + sense_range).min(self.map.height());

      Zip::indexed(person.brain.map.slice_mut(s![min_x..max_x, min_y..max_y]))
        .and(self.map.resources.slice(s![min_x..max_x, min_y..max_y]))
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
      let favorability = person.favorability_map();
      let dest = favorability
        .indexed_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
      let dx = dest.0 as isize - person.x as isize;
      let dy = dest.1 as isize - person.y as isize;

      if dx == 0 && dy == 0 {
        let consumed_cell =
          self.map.resources.get_mut((person.x, person.y)).unwrap();
        match consumed_cell {
          Resource::None => {}
          Resource::Food => {
            if person.needs.hunger >= HUNGER_PER_FOOD {
              person.needs.hunger -= HUNGER_PER_FOOD;
              *consumed_cell = Resource::None;
              person.brain.map[(person.x, person.y)] =
                ResourceProbability::probable(Resource::None, 1.0);
            }
          }
          Resource::Water => {
            if person.needs.thirst >= THIRST_PER_WATER {
              person.needs.thirst -= THIRST_PER_WATER;
              *consumed_cell = Resource::None;
              person.brain.map[(person.x, person.y)] =
                ResourceProbability::probable(Resource::None, 1.0);
            }
          }
          Resource::Stone => {}
          Resource::Ghost => {}
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

      let (a_i, a_share): (Vec<_>, Vec<_>) = a.brain.map.indexed_iter().unzip();
      let a_share: Vec<_> =
        a_i.into_iter().zip(a_share.into_iter().cloned()).collect();
      let b_share = b.brain.map.indexed_iter();

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
        self.map.resources.raw_dim(),
        collective_view
          .iter()
          .zip(self.map.resources.iter())
          .map(|(b, m)| b != m)
          .collect(),
      )
      .unwrap();

      for ((x, y), v) in error.indexed_iter() {
        let as_u8 = if *v { 255 } else { 0 };
        *buffer.get_pixel_mut(x as u32, y as u32) = Rgb([as_u8, as_u8, as_u8]);
      }
    }

    self.draw_selected_person(&mut img);

    img.into_inner()
  }

  fn draw_selected_person(&self, img: &mut ImageGrid) {
    let selected_person =
      if let Some(selected_person) = self.people.get(self.selected_person) {
        selected_person
      } else {
        return;
      };

    {
      let mut buffer = img.grid_mut(1, 0);

      selected_person.brain.draw(&mut buffer);
    }

    {
      let mut buffer = img.grid_mut(1, 1);

      let favorability = selected_person.favorability_map();
      let min = favorability.fold(0.0f64, |acc, cur| acc.min(*cur));
      let max = favorability.fold(0.0f64, |acc, cur| acc.max(*cur));

      for ((x, y), v) in selected_person.favorability_map().indexed_iter() {
        let as_u8 = (((*v - min) / (max - min)) * 255.0) as u8;
        *buffer.get_pixel_mut(x as u32, y as u32) = Rgb([as_u8, as_u8, as_u8]);
      }
    }

    {
      let mut buffer = img.grid_mut(2, 0);

      let error = Array2::from_shape_vec(
        self.map.resources.raw_dim(),
        selected_person
          .brain
          .map
          .map(|v| v.plurality())
          .iter()
          .zip(self.map.resources.iter())
          .map(|(b, m)| b != m)
          .collect(),
      )
      .unwrap();

      for ((x, y), v) in error.indexed_iter() {
        let as_u8 = if *v { 255 } else { 0 };
        *buffer.get_pixel_mut(x as u32, y as u32) = Rgb([as_u8, as_u8, as_u8]);
      }
    }
  }
}
