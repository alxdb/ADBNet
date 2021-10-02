{-# LANGUAGE FlexibleInstances #-}

module ADBNet.Tensor
  () where

import           Control.Applicative
import           Control.Monad
import qualified Data.Array.Unboxed            as U
import           Data.Ix
import           System.Random.Stateful

type R = Double

newtype Tensor a = Tensor { arr :: U.UArray a R }

type Matrix = Tensor (Int, Int)
type Vector = Tensor Int
type Scalar = Tensor ()

instance Show Matrix where
  show = show . rows

instance Show Vector where
  show = show . U.elems . arr

instance Show Scalar where
  show = show . (! ())

class Ix a => Tix a where
  start :: a
  tRange :: a -> (a, a)
  tRange s = (start, s)

instance Tix (Int, Int) where
  start = (1, 1)

instance Tix Int where
  start = 1

instance Tix () where
  start = ()

tMap :: (Ix a) => (R -> R) -> Tensor a -> Tensor a
tMap f = Tensor . U.amap f . arr

tZip :: (Ix a) => (R -> R -> R) -> Tensor a -> Tensor a -> Tensor a
tZip f a b | tDim a == tDim b = undefined
           | otherwise        = error "Tensor dimensions do not match"

tDim :: (Ix a) => Tensor a -> a
tDim = snd . U.bounds . arr

tNew :: (Tix a) => a -> [R] -> Tensor a
tNew s v = Tensor $ U.listArray (tRange s) v

tRndM :: (Tix a, StatefulGen g m) => a -> g -> m (Tensor a)
tRndM s g =
  tNew s <$> replicateM (rangeSize . tRange $ s) (uniformRM (0.0, 1.0) g)

tRnd :: (Tix a, RandomGen g) => a -> g -> (Tensor a, g)
tRnd s g = runStateGen g (tRndM s)

(!) :: (Ix a) => Tensor a -> a -> R
(!) = (U.!) . arr

row :: Matrix -> Int -> Vector
row m r = tNew c $ map (\i -> m ! (r, i)) [1 .. c] where c = snd . tDim $ m

rows :: Matrix -> [Vector]
rows m = map (row m) [1 .. (fst . tDim $ m)]

col :: Matrix -> Int -> Vector
col m c = tNew r $ map (\i -> m ! (i, c)) [1 .. r] where r = fst . tDim $ m

cols :: Matrix -> [Vector]
cols m = map (col m) [1 .. (snd . tDim $ m)]
