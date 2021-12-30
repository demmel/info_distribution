use enum_ordinalize::Ordinalize;

use crate::graphics::Color;

#[derive(Clone, Copy, PartialEq, Eq, Ordinalize)]
pub(crate) enum Resource {
  None,
  Food,
  Water,
  Stone,
  Ghost,
}

impl Resource {
  pub(crate) fn color(&self) -> Color {
    match self {
      Resource::None => [0.0, 0.0, 0.0],
      Resource::Food => [0.0, 1.0, 0.0],
      Resource::Water => [0.0, 0.0, 1.0],
      Resource::Stone => [0.5, 0.5, 0.5],
      Resource::Ghost => [0.75, 0.3, 0.75],
    }
    .into()
  }
}
