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

type Network = Vector (Matrix Double)

type Activations = Vector (Vector Double, Vector Double)

randNet :: (StatefulGen g m) => [Int] -> g -> m Network
randNet dims g = V.fromList <$> mapM (`randMat` g) (zipWith (\i o -> (o, i)) dims (tail dims))

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp x)

sigmoid' :: Floating a => a -> a
sigmoid' x = sigmoid x * (1 - sigmoid x)

rss :: Floating a => a -> a -> a
rss x y = (x - y) ** 2

rss' :: Floating a => a -> a -> a
rss' x y = 2 * (x - y)

memoize :: Int -> (Int -> a) -> (Int -> a)
memoize l f = (V.generate l f V.!)

activations :: Network -> Vector Double -> Activations
activations net input = V.generate (length net) (activation net input)
  where
    activation net input = fix (memoize (length net) . f)
      where
        f _ 0 = trace "activate 0" $ let z = V.head net !* input in (V.map sigmoid z, z)
        f f' n = trace ("activate " ++ show n) $ let z = net V.! n !* fst (f' (n - 1)) in (V.map sigmoid z, z)

deltas :: Network -> Vector Double -> Vector Double -> Activations -> Vector (Vector Double)
deltas net input output as = V.generate (length net) (delta net input output as)
  where
    delta net input output as = fix (memoize (length net) . f)
      where
        f f' n
          | n == length net - 1 = trace ("delta " ++ show n) $ V.zipWith (*) z' (V.zipWith rss' a output)
          | otherwise = trace ("delta " ++ show n) $ V.zipWith (*) z' (c !* f' (n + 1))
          where
            (a, z) = as V.! n
            c = net V.! n
            z' = V.map sigmoid' z

gradients :: Network -> Vector Double -> Vector Double -> Activations -> Vector (Vector Double) -> Vector (Matrix Double)
gradients net input output as ds = V.generate (length net) (gradient net input output as ds)
  where
    gradient net input output as ds = fix (memoize (length net) . f)
      where
        f f' 0 = trace "gradient 0" $ outer input (V.head ds)
        f f' n = trace ("gradient " ++ show n) $ outer (fst $ as V.! (n - 1)) (ds V.! n)

main :: IO ()
main = do
  gen <- newIOGenM $ mkStdGen 10
  net <- randNet [2, 4, 4, 1] gen
  let input = [0, 0]
  let output = [0]
  let as = activations net input
  print $ gradients net input output as (deltas net input output as)
