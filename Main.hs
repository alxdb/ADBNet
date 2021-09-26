{-# LANGUAGE OverloadedLists #-}

module Main where

import Control.Monad
import Data.Function
import qualified Data.Vector as V
import Debug.Trace
import Numeric.LinearAlgebra
import System.Random.Stateful

randMat :: (StatefulGen g m) => (Int, Int) -> g -> m (Matrix R)
randMat (rows, cols) g = (rows >< cols) <$> replicateM (rows * cols) (uniformRM (0.0, 1.0) g)

type Network = V.Vector (Matrix R)

randNet :: (StatefulGen g m) => [Int] -> g -> m Network
randNet dims g = V.fromList <$> mapM (`randMat` g) (zipWith (\i o -> (o, i + 1)) dims (tail dims))

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

type Activations = V.Vector (Vector R, Vector R)

activations :: Network -> Vector R -> Activations
activations net input = V.generate (length net) (activation net input)
  where
    activation :: Network -> Vector R -> Int -> (Vector R, Vector R)
    activation net input = fix (memoize (length net) . f)
      where
        f :: (Int -> (Vector R, Vector R)) -> Int -> (Vector R, Vector R)
        f _ 0 = trace "activate 0" $ let z = V.head net #> vjoin [input, 1.0] in (sigmoid z, z)
        f f' n = trace ("activate " ++ show n) $ let z = net V.! n #> vjoin [fst (f' (n - 1)), 1.0] in (sigmoid z, z)

deltas :: Network -> Vector R -> Vector R -> Activations -> V.Vector (Vector R)
deltas net input output as = V.generate (length net) (delta net input output as)
  where
    delta :: Network -> Vector R -> Vector R -> Activations -> Int -> Vector R
    delta net input output as = fix (memoize (length net) . f)
      where
        f :: (Int -> Vector R) -> Int -> Vector R
        f f' n
          | n == length net - 1 = trace ("delta " ++ show n) $ z' * rss' a output
          | otherwise = trace ("delta " ++ show n) $ z' * (tr w #> f' (n + 1))
          where
            (a, z) = as V.! n
            w = net V.! n ¿ [0 .. size a - 2]
            z' = sigmoid' z

gradients :: Network -> Vector R -> Vector R -> Activations -> V.Vector (Vector R) -> V.Vector (Matrix R)
gradients net input output as ds = V.generate (length net) (gradient net input output as ds)
  where
    gradient :: Network -> Vector R -> Vector R -> Activations -> V.Vector (Vector R) -> Int -> Matrix R
    gradient net input output as ds = fix (memoize (length net) . f)
      where
        f f' 0 = trace "gradient 0" $ input `outer` V.head ds
        f f' n = trace ("gradient " ++ show n) $ let (a, _) = as V.! (n - 1) in a `outer` (ds V.! n)

main = do
  gen <- newIOGenM $ mkStdGen 10
  net <- randNet [2, 4, 4, 1] gen
  let input = [0, 0]
  let output = [0]
  let as = activations net input in print $ gradients net input output as (deltas net input output as)
