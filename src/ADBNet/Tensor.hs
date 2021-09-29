{-# LANGUAGE FlexibleInstances #-}

module ADBNet.Tensor
  () where

import           Control.Applicative
import           Control.Monad
import           Data.Array.Unboxed
import           System.Random.Stateful

type R = Double

newtype Tensor a = Tensor { arr :: UArray a R }

type Matrix = Tensor (Int, Int)
type Vector = Tensor Int

instance Show Matrix where
  show = show . arr

instance Show Vector where
  show = show . elems . arr

class Ix a => Tix a where
  start :: a
  tRange :: a -> (a, a)
  tRange s = (start, s)

instance Tix (Int, Int) where
  start = (1, 1)

instance Tix Int where
  start = 1

tMap :: (Ix a) => (R -> R) -> Tensor a -> Tensor a
tMap f = Tensor . amap f . arr

tDim :: (Ix a) => Tensor a -> a
tDim = snd . bounds . arr

tNew :: (Tix a) => a -> [R] -> Tensor a
tNew s v = Tensor $ listArray (tRange s) v

tRndM :: (Tix a, StatefulGen g m) => a -> g -> m (Tensor a)
tRndM s g =
  tNew s <$> replicateM (rangeSize . tRange $ s) (uniformRM (0.0, 1.0) g)

tRnd :: (Tix a, RandomGen g) => a -> g -> (Tensor a, g)
tRnd s g = runStateGen g (tRndM s)
