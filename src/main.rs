use image::{GenericImage, ImageBuffer, Rgb, RgbImage, SubImage};
use ndarray::{s, Array2};
use rand::{thread_rng, Rng};
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

  let map = Map::gen_map(100, 100);

  let people: Vec<_> = (0..100)
    .map(|_| Person {
      brain: Map::gen_brain(map.width(), map.height()),
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
    for person in self.people.iter_mut() {
      let sense_range = 5;
      let min_x = person.x.saturating_sub(sense_range);
      let max_x = (person.x + sense_range).min(self.map.width());

      let min_y = person.y.saturating_sub(sense_range);
      let max_y = (person.y + sense_range).min(self.map.height());

      person
        .brain
        .map
        .slice_mut(s![min_x..max_x, min_y..max_y])
        .zip_mut_with(
          &self.map.map.slice(s![min_x..max_x, min_y..max_y]),
          |b, m| {
            *b = (*b + *m) / 2.0;
          },
        );
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
            Rgb([0, 255, 0])
          } else {
            Rgb([255, 0, 0])
          };
      }
    }

    {
      let mut buffer = img.grid_mut(0, 1);

      let mut mse = self
        .people
        .iter()
        .map(|p| {
          let mut e = &p.brain.map - &self.map.map;
          e.map_inplace(|v| *v = *v * *v);
          e
        })
        .fold(
          Array2::<f64>::zeros(self.map.map.raw_dim()),
          |mut acc, cur| {
            acc += &cur;
            acc
          },
        );
      mse /= self.people.len() as f64;

      for ((x, y), v) in mse.indexed_iter() {
        let as_u8 = (v * 255.0) as u8;
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

      let error = (&selected_person.brain.map - &self.map.map).map(|v| v.abs());

      for ((x, y), v) in error.indexed_iter() {
        let as_u8 = (v * 255.0) as u8;
        *buffer.get_pixel_mut(x as u32, y as u32) = Rgb([as_u8, as_u8, as_u8]);
      }
    }

    img.into_inner()
  }
}

struct Map {
  map: Array2<f64>,
}

impl Map {
  fn gen_map(width: usize, height: usize) -> Self {
    let mut rng = rand::thread_rng();
    Self {
      map: Array2::from_shape_simple_fn((width, height), || {
        if rng.gen() {
          1.0
        } else {
          0.0
        }
      }),
    }
  }

  fn gen_brain(width: usize, height: usize) -> Self {
    let mut rng = rand::thread_rng();
    Self {
      map: Array2::from_shape_simple_fn((width, height), || rng.gen()),
    }
  }

  fn width(&self) -> usize {
    self.map.shape()[0]
  }

  fn height(&self) -> usize {
    self.map.shape()[1]
  }

  fn draw(&self, img: &mut SubImage<&mut RgbImage>) {
    for ((x, y), v) in self.map.indexed_iter() {
      let as_u8 = (v * 255.0) as u8;
      *img.get_pixel_mut(x as u32, y as u32) = Rgb([as_u8, as_u8, as_u8]);
    }
  }
}

struct Person {
  brain: Map,
  x: usize,
  y: usize,
}

struct ImageGrid {
  img: RgbImage,
  width: usize,
  height: usize,
}

impl ImageGrid {
  fn new(width: usize, height: usize, c: usize, r: usize) -> Self {
    let img: RgbImage = ImageBuffer::new(
      (c * width + (c - 1)) as u32,
      (r * height + (r - 1)) as u32,
    );
    Self { img, width, height }
  }

  fn into_inner(self) -> RgbImage {
    self.img
  }

  fn grid_mut(&mut self, c: usize, r: usize) -> SubImage<&mut RgbImage> {
    self.img.sub_image(
      (c * (self.width + 1)) as u32,
      (r * (self.height + 1)) as u32,
      self.width as u32,
      self.height as u32,
    )
  }
}
