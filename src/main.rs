use image::{ImageBuffer, Rgb, RgbImage};
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
  let map = Map::gen_map(200, 100);

  let mut people: Vec<_> = (0..50)
    .map(|_| Person {
      brain: Map::gen_brain(200, 100),
      x: rng.gen_range(0..map.width()),
      y: rng.gen_range(0..map.height()),
    })
    .collect();
  let mut selected_person = 0;

  let map_window = create_window(
    "Map",
    WindowOptions {
      size: Some([1000, 1000]),
      ..Default::default()
    },
  )?;
  map_window.set_image("map", map.as_image())?;

  let people_map_window = create_window(
    "People Map",
    WindowOptions {
      size: Some([1000, 1000]),
      ..Default::default()
    },
  )?;
  people_map_window
    .set_image("people", map_of_people(&map, &people, selected_person))?;

  let brain_window = create_window(
    "Brain",
    WindowOptions {
      size: Some([1000, 1000]),
      ..Default::default()
    },
  )?;
  brain_window.set_image("brain", people[selected_person].brain.as_image())?;

  let mse_window = create_window(
    "MSE",
    WindowOptions {
      size: Some([1000, 1000]),
      ..Default::default()
    },
  )?;
  mse_window.set_image("mse", mse_of_brains(&map, &people))?;

  for event in brain_window.event_channel()? {
    if let WindowEvent::KeyboardInput(event) = event {
      if !event.input.state.is_pressed() {
        continue;
      }
      match event.input.key_code {
        event @ Some(VirtualKeyCode::Left | VirtualKeyCode::Right) => {
          match event {
            Some(VirtualKeyCode::Left) => {
              selected_person =
                selected_person.checked_sub(1).unwrap_or(people.len() - 1);
            }
            Some(VirtualKeyCode::Right) => {
              selected_person = (selected_person + 1) % people.len();
            }
            _ => {
              unreachable!()
            }
          };

          brain_window
            .set_image("brain", people[selected_person].brain.as_image())?;
          people_map_window.set_image(
            "people",
            map_of_people(&map, &people, selected_person),
          )?;
        }
        Some(VirtualKeyCode::Space) => {
          for person in people.iter_mut() {
            let sense_range = 5;
            let min_x = person.x.saturating_sub(sense_range);
            let max_x = (person.x + sense_range).min(map.width());

            let min_y = person.y.saturating_sub(sense_range);
            let max_y = (person.y + sense_range).min(map.height());

            person
              .brain
              .map
              .slice_mut(s![min_x..max_x, min_y..max_y])
              .zip_mut_with(
                &map.map.slice(s![min_x..max_x, min_y..max_y]),
                |b, m| {
                  *b = (*b + *m) / 2.0;
                },
              );
          }
          brain_window
            .set_image("brain", people[selected_person].brain.as_image())?;
          mse_window.set_image("mse", mse_of_brains(&map, &people))?;
        }
        _ => {}
      }
    }
  }

  Ok(())
}

fn map_of_people(
  map: &Map,
  people: &[Person],
  selected_person: usize,
) -> RgbImage {
  let mut img: RgbImage =
    ImageBuffer::new(map.width() as u32, map.height() as u32);

  for (i, person) in people.iter().enumerate() {
    *img.get_pixel_mut(person.x as u32, person.y as u32) =
      if selected_person == i {
        Rgb([255, 255, 255])
      } else {
        Rgb([255, 0, 0])
      };
  }

  img
}

fn mse_of_brains(map: &Map, people: &[Person]) -> RgbImage {
  let mut mse = people
    .iter()
    .map(|p| {
      let mut e = &p.brain.map - &map.map;
      e.map_inplace(|v| *v = *v * *v);
      e
    })
    .fold(Array2::<f64>::zeros(map.map.raw_dim()), |mut acc, cur| {
      acc += &cur;
      acc
    });
  mse /= people.len() as f64;

  let mut img: RgbImage =
    ImageBuffer::new(mse.shape()[0] as u32, mse.shape()[1] as u32);

  for (x, y, pixel) in img.enumerate_pixels_mut() {
    let v = mse[(x as usize, y as usize)];
    let as_u8 = (v * 255.0) as u8;

    *pixel = Rgb([as_u8, as_u8, as_u8]);
  }

  img
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

  fn as_image(&self) -> RgbImage {
    let mut img: RgbImage =
      ImageBuffer::new(self.width() as u32, self.height() as u32);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
      let v = self.map[(x as usize, y as usize)];
      let as_u8 = (v * 255.0) as u8;

      *pixel = Rgb([as_u8, as_u8, as_u8]);
    }

    img
  }
}

struct Person {
  brain: Map,
  x: usize,
  y: usize,
}

impl Person {}
