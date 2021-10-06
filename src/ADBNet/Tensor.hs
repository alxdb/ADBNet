{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ConstraintKinds #-}

module ADBNet.Tensor
  ( Tensor
  , Matrix
  , Vector
  , Tix
  , ixs
  , size
  , dims
  , tmap
  , tzip
  , tnew
  , tval
  , trndM
  , trnd
  , trnd_
  , scale
  , mnew
  , vnew
  , vapp
  , row
  , rows
  , fromRows
  , col
  , cols
  , fromCols
  , addCol
  , remCol
  , matmul
  , outerp
  , transp
  ) where

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

class (Show i, Ix i) => Tix i where
  ixs :: i -> (i, i)
  size :: i -> Int

instance Tix Int where
  ixs d = (1, d)
  size d = d

instance Tix (Int, Int) where
  ixs d = ((1, 1), d)
  size d = uncurry (*) d

type El e = (IArray UArray e)

instance (Show e, El e) => Show (Matrix e) where
  show = show . rows

instance (Show e, El e) => Show (Vector e) where
  show = show . elems

instance (El e) => IArray Tensor e where
  bounds           = bounds . arr
  numElements      = numElements . arr
  unsafeArray      = \a b -> Tensor $ unsafeArray a b
  unsafeAt         = unsafeAt . arr
  unsafeReplace    = \a b -> Tensor $ unsafeReplace (arr a) b
  unsafeAccum      = \a b c -> Tensor $ unsafeAccum a (arr b) c
  unsafeAccumArray = \a b c d -> Tensor $ unsafeAccumArray a b c d

instance (Ix i, El e, Eq e) => Eq (Tensor i e) where
  a /= b = arr a /= arr b
  a == b = arr a == arr b

instance (Ix i, El e, Ord e) => Ord (Tensor i e) where
  compare a b = compare (arr a) (arr b)

instance (Tix i, Show i, El e, Num e) => Num (Tensor i e) where
  (+)         = tzip (+)
  (*)         = tzip (*)
  abs         = tmap abs
  signum      = tmap signum
  negate      = tmap negate
  fromInteger = error "Sorry! Can't use fromInteger with Tensors"

instance (Tix i, El e, Fractional e) => Fractional (Tensor i e) where
  (/)          = tzip (/)
  recip        = tmap recip
  fromRational = error "Sorry! Can't use fromRational with Tensors"

instance (Tix i, El e, Floating e) => Floating (Tensor i e) where
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

dims :: (Tix i, El e) => Tensor i e -> i
dims = snd . bounds

tmap :: (Tix i, El e, El f) => (e -> f) -> Tensor i e -> Tensor i f
tmap f = Tensor . amap f . arr

tzip
  :: (Tix i, El e, El f, El g)
  => (e -> f -> g)
  -> Tensor i e
  -> Tensor i f
  -> Tensor i g
tzip f a b
  | dims a == dims b
  = listArray (bounds a) $ zipWith f (elems a) (elems b)
  | otherwise
  = error
    $  "Tensor dimensions do not match: "
    ++ show (dims a)
    ++ " != "
    ++ show (dims b)

tnew :: (Tix i, El e) => i -> [e] -> Tensor i e
tnew d v = Tensor $ listArray (ixs d) v

tval :: (Tix i, El e) => i -> e -> Tensor i e
tval d v = tnew d (repeat v)

trndM
  :: (Tix i, El e, UniformRange e, StatefulGen g m)
  => i
  -> (e, e)
  -> g
  -> m (Tensor i e)
trndM s r g = tnew s <$> replicateM (rangeSize . ixs $ s) (uniformRM r g)

trnd
  :: (Tix i, El e, UniformRange e, RandomGen g)
  => i
  -> (e, e)
  -> g
  -> (Tensor i e, g)
trnd s r g = runStateGen g (trndM s r)

trnd_ :: (Tix i, El e, UniformRange e) => i -> (e, e) -> Int -> Tensor i e
trnd_ s r g = runStateGen_ (mkStdGen g) (trndM s r)

scale :: (Tix i, El e, Num e) => e -> Tensor i e -> Tensor i e
scale x = tmap (* x)

mnew :: (El e) => (Int, Int) -> [e] -> Matrix e
mnew = tnew

vnew :: (El e) => Int -> [e] -> Vector e
vnew = tnew

foldv :: (El e) => (a -> e -> a) -> a -> Vector e -> a
foldv f a = foldl f a . elems

sumv :: (El e, Num e) => Vector e -> e
sumv = foldv (+) 0

dotv :: (El e, Num e) => Vector e -> Vector e -> e
dotv u = sumv . (* u)

vapp :: (El e, Num e) => Vector e -> e -> Vector e
vapp v x = tnew (dims v + 1) $ elems v ++ [x]

row :: (El e) => Matrix e -> Int -> Vector e
row m r = tnew c $ map (\i -> m ! (r, i)) [1 .. c] where c = snd . dims $ m

rows :: (El e) => Matrix e -> [Vector e]
rows m = map (row m) [1 .. (fst . dims $ m)]

fromRows :: (El e) => [Vector e] -> Matrix e
fromRows rs = tnew (length rs, dims . head $ rs) $ concatMap elems rs

col :: (El e) => Matrix e -> Int -> Vector e
col m c = tnew r $ map (\i -> m ! (i, c)) [1 .. r] where r = fst . dims $ m

cols :: (El e) => Matrix e -> [Vector e]
cols m = map (col m) [1 .. (snd . dims $ m)]

fromCols :: (El e) => [Vector e] -> Matrix e
fromCols = transp . fromRows

addCol :: (El e) => Matrix e -> Vector e -> Matrix e
addCol m c = fromCols $ cols m ++ [c]

remCol :: (El e) => Matrix e -> Matrix e
remCol m = fromCols . init $ cols m

matmul :: (El e, Num e, Enum e) => Matrix e -> Vector e -> Vector e
matmul m v = vnew (dims v) $ map (dotv v) (rows m)

outerp :: (El e, Num e) => Vector e -> Vector e -> Matrix e
outerp u v = fromRows . map (`scale` v) $ elems u

transp :: (El e) => Matrix e -> Matrix e
transp = fromRows . cols
