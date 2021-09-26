{-# LANGUAGE OverloadedLists #-}

module Main where

import Data.Function
import qualified Data.Vector as V
import Debug.Trace
import Linear hiding (trace)
import System.Random.Stateful

type Vector a = V.Vector a

type Matrix a = Vector (Vector a)

randMat :: (UniformRange a, Fractional a, StatefulGen g m) => (Int, Int) -> g -> m (Matrix a)
randMat (rows, cols) g = V.replicateM rows . V.replicateM cols $ uniformRM (0.0, 1.0) g

type Network = (Vector (Matrix Double))

type Activations = (Vector (Vector Double, Vector Double))

randNet :: (StatefulGen g m) => [Int] -> g -> m Network
randNet dims g = V.fromList <$> mapM (`randMat` g) (zipWith (\i o -> (o, i)) dims (tail dims))

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp x)

memoize :: Int -> (Int -> a) -> (Int -> a)
memoize l f = (V.generate l f V.!)

activation :: Network -> Vector Double -> Int -> (Vector Double, Vector Double)
activation net input = fix (memoize (length net) . f)
  where
    f _ 0 = trace "activate 0" $ let z = V.head net !* input in (V.map sigmoid z, z)
    f f' n = trace ("activate " ++ show n) $ let z = net V.! n !* fst (f' (n - 1)) in (V.map sigmoid z, z)

activations :: Network -> Vector Double -> Activations
activations net input = V.generate (length net) (activation net input)

delta :: Network -> Vector Double -> Vector Double -> Int -> Vector Double
delta net input output = fix (memoize (length net) . f)
  where
    f f' n
      | n == 0 = undefined
      | n == length net - 1 = undefined
      | otherwise = undefined

main :: IO ()
main = do
  gen <- newIOGenM $ mkStdGen 10
  net <- randNet [2, 4, 4, 1] gen
  print $ activations net [0, 0]
