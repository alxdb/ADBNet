module ADBNet.Maths
  ( Matrix
  , Vector
  , matrix
  , randomMatrix
  ) where

import           Control.Monad
import           Data.Array.Unboxed
import           System.Random.Stateful

newtype Matrix = Matrix (UArray (Int, Int) Double) deriving (Show)
newtype Vector = Vector (UArray Int Double) deriving (Show)

matrix :: (Int, Int) -> [Double] -> Matrix
matrix size@(rows, cols) values =
  Matrix $ array ((0, 0), (rows - 1, cols - 1)) $ zip index values
  where index = [ (i, j) | i <- [0 .. rows - 1], j <- [0 .. cols - 1] ]

randomMatrix :: StatefulGen g m => (Int, Int) -> g -> m Matrix
randomMatrix size@(rows, cols) g = do
  values <- replicateM (rows * cols) $ uniformRM (0.0, 1.0) g
  return $ matrix size values
