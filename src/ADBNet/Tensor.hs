{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}

module ADBNet.Tensor
  () where

import           Control.Applicative
import           Control.Monad
import           Data.Array.Base
import           Data.Array.Unboxed
import           Data.Ix
import           System.Random.Stateful
import           Text.Printf

newtype Tensor i e = Tensor { arr :: UArray i e }

type Matrix e = Tensor (Int, Int) e
type Vector e = Tensor Int e
type Scalar e = Tensor () e

class (Show i, Ix i) => Tix i where
  ixs :: i -> (i, i)

instance Tix () where
  ixs _ = ((), ())

instance Tix Int where
  ixs d = (1, d)

instance Tix (Int, Int) where
  ixs d = ((1, 1), d)

instance (IArray UArray e) => IArray Tensor e where
  bounds           = bounds . arr
  numElements      = numElements . arr
  unsafeArray      = \a b -> Tensor $ unsafeArray a b
  unsafeAt         = unsafeAt . arr
  unsafeReplace    = \a b -> Tensor $ unsafeReplace (arr a) b
  unsafeAccum      = \a b c -> Tensor $ unsafeAccum a (arr b) c
  unsafeAccumArray = \a b c d -> Tensor $ unsafeAccumArray a b c d

instance (Ix i, IArray UArray e, Eq e) => Eq (Tensor i e) where
  a /= b = arr a /= arr b
  a == b = arr a == arr b

instance (Ix i, IArray UArray e, Ord e) => Ord (Tensor i e) where
  compare a b = compare (arr a) (arr b)

instance (Tix i, Show i, IArray UArray e, Num e) => Num (Tensor i e) where
  (+)         = tzip (+)
  (*)         = tzip (*)
  abs         = tmap abs
  signum      = tmap signum
  negate      = tmap negate
  fromInteger = error "Sorry! Can't use fromInteger with Tensors"

instance (Tix i, IArray UArray e, Fractional e) => Fractional (Tensor i e) where
  (/)          = tzip (/)
  recip        = tmap recip
  fromRational = error "Sorry! Can't use fromRational with Tensors"

instance (Tix i, IArray UArray e, Floating e) => Floating (Tensor i e) where
  pi    = error "Sorry! Can't use pi with Tensors"
  exp   = tmap exp
  log   = tmap log
  sin   = tmap sin
  cos   = tmap cos
  asin  = tmap asin
  acos  = tmap acos
  atan  = tmap atan
  sinh  = tmap sinh
  cosh  = tmap cosh
  asinh = tmap asinh
  acosh = tmap acosh
  atanh = tmap atanh

dims :: (Tix i, IArray UArray e) => Tensor i e -> i
dims = snd . bounds

tmap :: (Tix i, IArray UArray e) => (e -> e) -> Tensor i e -> Tensor i e
tmap f = Tensor . amap f . arr

tzip
  :: (Tix i, IArray UArray e)
  => (e -> e -> e)
  -> Tensor i e
  -> Tensor i e
  -> Tensor i e
tzip f a b
  | dims a == dims b
  = listArray (bounds a) $ zipWith f (elems a) (elems b)
  | otherwise
  = error
    $  "Tensor dimensions do not match: "
    ++ show (dims a)
    ++ " != "
    ++ show (dims b)

tnew :: (Tix i, IArray UArray e) => i -> [e] -> Tensor i e
tnew d v = Tensor $ listArray (ixs d) v

trndM
  :: (Tix i, IArray UArray e, UniformRange e, StatefulGen g m)
  => i
  -> (e, e)
  -> g
  -> m (Tensor i e)
trndM s r g = tnew s <$> replicateM (rangeSize . ixs $ s) (uniformRM r g)

trnd
  :: (Tix i, IArray UArray e, UniformRange e, RandomGen g)
  => i
  -> (e, e)
  -> g
  -> (Tensor i e, g)
trnd s r g = runStateGen g (trndM s r)

instance (Show e, IArray UArray e) => Show (Matrix e) where
  show = show . rows

instance (Show e, IArray UArray e) => Show (Vector e) where
  show = show . elems

instance (Show e, IArray UArray e) => Show (Scalar e) where
  show = show . (! ())

row :: (IArray UArray e) => Matrix e -> Int -> Vector e
row m r = tnew c $ map (\i -> m ! (r, i)) [1 .. c] where c = snd . dims $ m

rows :: (IArray UArray e) => Matrix e -> [Vector e]
rows m = map (row m) [1 .. (fst . dims $ m)]

col :: (IArray UArray e) => Matrix e -> Int -> Vector e
col m c = tnew r $ map (\i -> m ! (i, c)) [1 .. r] where r = fst . dims $ m

cols :: (IArray UArray e) => Matrix e -> [Vector e]
cols m = map (col m) [1 .. (snd . dims $ m)]

matmul :: (IArray UArray e, Num e, Enum e) => Matrix e -> Vector e -> Vector e
matmul m v = foldl (+) (tnew (dims v) [0 ..]) (map (* v) $ rows m)
