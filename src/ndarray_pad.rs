use ndarray::{s, Array2, IntoDimension, Ix2};

pub enum ArrayPaddingKind<T: Clone> {
  Constant(T),
  Clamp,
}

pub trait ArrayPaddingExt<T: Clone> {
  fn pad<PS>(
    &self,
    padding_size: PS,
    padding_kind: ArrayPaddingKind<T>,
  ) -> Array2<T>
  where
    PS: IntoDimension<Dim = Ix2>;
}

impl<T: Clone> ArrayPaddingExt<T> for Array2<T> {
  fn pad<PS>(
    &self,
    padding_size: PS,
    padding_kind: ArrayPaddingKind<T>,
  ) -> Array2<T>
  where
    PS: IntoDimension<Dim = Ix2>,
  {
    let padding_size = padding_size.into_dimension();
    let dim = self.raw_dim() + (padding_size * 2);

    match padding_kind {
      ArrayPaddingKind::Constant(t) => {
        let mut arr = Array2::from_elem(dim, t);
        arr
          .slice_mut(s![
            padding_size[0]..(dim[0] - padding_size[0]),
            padding_size[1]..(dim[1] - padding_size[1])
          ])
          .assign(self);
        arr
      }
      ArrayPaddingKind::Clamp => Array2::from_shape_fn(dim, |(x, y)| {
        let x = x.clamp(padding_size[0], dim[0] - padding_size[0] - 1)
          - padding_size[0];
        let y = y.clamp(padding_size[1], dim[1] - padding_size[1] - 1)
          - padding_size[1];
        self[(x, y)].clone()
      }),
    }
  }
}
