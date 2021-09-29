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
  trange :: a -> (a, a)
  trange s = (start, s)

instance Tix (Int, Int) where
  start = (1, 1)

instance Tix Int where
  start = 1

tmap :: (Ix a) => (R -> R) -> Tensor a -> Tensor a
tmap f = Tensor . amap f . arr

tdim :: (Ix a) => Tensor a -> a
tdim = snd . bounds . arr

tnew :: (Tix a) => a -> [R] -> Tensor a
tnew s v = Tensor $ listArray (trange s) v

trnd :: (Tix a, StatefulGen g m) => a -> g -> m (Tensor a)
trnd s g =
  tnew s <$> replicateM (rangeSize . trange $ s) (uniformRM (0.0, 1.0) g)

-- class ITensor t a where
--   tnew :: a -> [R] -> t a

-- instance ITensor Tensor (Int, Int) where
--   tnew s v = Tensor $ listArray ((1, 1), s) v

-- instance ITensor Tensor Int where
--   tnew s v = Tensor $ listArray (1, s) v

-- mrnd :: StatefulGen g m => (Int, Int) -> g -> m Matrix
-- mrnd s@(r, c) g = tnew . array

-- type family Rank t
-- type instance Rank Matrix = (Int, Int)
-- type instance Rank Vector = Int

-- newtype Matrix = Matrix (UArray (Rank Matrix) R)
-- newtype Vector = Vector (UArray (Rank Vector) R)


-- class Tensor t where
--   toArray :: t -> UArray (Rank t) R
--   fromArray :: UArray (Rank t) R -> t

-- instance Tensor Matrix where
--   toArray (Matrix arr) = arr
--   fromArray arr = Matrix arr

-- instance Tensor Vector where
--   toArray (Vector arr) = arr
--   fromArray arr = Vector arr

  -- dim :: t -> Rank t
  -- dim = snd . bounds . toArray

  -- new :: Rank t -> [R] -> t
  -- rnd :: StatefulGen g m => Rank t -> g -> m t

  -- map f (Matrix arr) = Matrix $ amap f arr
  -- dim (Matrix arr) = let (x, y) = fst . bounds $ arr in (x + 1, y + 1)
  -- new size@(rows, cols) v = Matrix . array b $ zip i v
  --  where
  --   r = rows - 1
  --   c = cols - 1
  --   b = ((0, 0), (r, c))
  --   i = [ (i, j) | i <- [0 .. r], j <- [0 .. c] ]
  -- rnd size@(rows, cols) g = do
  --   values <- replicateM (rows * cols) $ uniformRM (0.0, 1.0) g
  --   return $ new size values

  -- map f (Vector arr) = Vector $ amap f arr
  -- dim (Vector arr) = let x = fst . bounds $ arr in x + 1
  -- new length v = Vector . array b $ zip i v
  --  where
  --   l = length - 1
  --   b = (0, l)
  --   i = [0 .. l]
  -- rnd length v = do
  --   values <- replicateM length $ uniformRM (0.0, 1.0) g
  --   return $ new length values
