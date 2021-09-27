{-# LANGUAGE OverloadedLists #-}

module Main where

import Control.Monad
import Data.Function
import qualified Data.Vector as V
import Numeric.LinearAlgebra
import System.Random.Stateful
import Text.Printf

-- | To be replaced with Debug.Trace when debugging
trace :: String -> b -> b
trace _ x = x

randMat :: (StatefulGen g m) => (Int, Int) -> g -> m (Matrix R)
randMat (rows, cols) g = (rows >< cols) <$> replicateM (rows * cols) (uniformRM (0.0, 1.0) g)

type Network = V.Vector (Matrix R)

randNet :: (StatefulGen g m) => [Int] -> g -> m Network
randNet dims g = V.fromList <$> mapM (`randMat` g) (zipWith (\i o -> (o, i + 1)) dims (tail dims))

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (- x))

sigmoid' :: Floating a => a -> a
sigmoid' x = sigmoid x * (1 - sigmoid x)

rss :: Floating a => a -> a -> a
rss x y = (x - y) ** 2

rss' :: Floating a => a -> a -> a
rss' x y = 2 * (x - y)

type Activations = V.Vector (Vector R, Vector R)

activations :: Network -> Vector R -> Activations
activations net input = result
  where
    result = V.generate (length net) f
    f 0 = trace "a 0" $ let z = V.head net #> vjoin [input, 1.0] in (sigmoid z, z)
    f n = trace (printf "a %d" n) $ let z = net V.! n #> vjoin [fst (result V.! (n - 1)), 1.0] in (sigmoid z, z)

run :: Network -> Vector R -> Vector R
run net input = fst . V.last $ activations net input

deltas :: Network -> Vector R -> Vector R -> Activations -> V.Vector (Vector R)
deltas net input output as = result
  where
    result = V.generate (length net) f
    f n = trace (printf "d %d" n) $ z' * if n == length net - 1 then rss' a output else tr w #> result V.! (n + 1)
      where
        (a, z) = as V.! n
        c = net V.! (n + 1)
        w = takeColumns (cols c - 1) c
        z' = sigmoid' z

gradients :: Network -> Vector R -> Vector R -> Activations -> V.Vector (Vector R) -> V.Vector (Matrix R)
gradients net input output as ds = V.generate (length net) f
  where
    f 0 = trace "g 0" $ V.head ds `outer` vjoin [input, 1.0]
    f n = trace (printf "g %d" n) $ (ds V.! n) `outer` vjoin [fst (as V.! (n - 1)), 1.0]

fit :: Network -> R -> Vector R -> Vector R -> Activations -> Network
fit net lr input output as = V.zipWith (-) net gs
  where
    gs = V.map (cmap (lr *)) (gradients net input output as ds)
    ds = deltas net input output as

train :: Network -> R -> [(Vector R, Vector R)] -> Network
train net lr trainingData = fst $ foldl f (net, as') trainingData
  where
    as' = activations net (fst . head $ trainingData)
    f (n, as) (i, o) = let n' = fit n lr i o as in (n', activations n' i)

test :: Network -> [(Vector R, Vector R)] -> [R]
test net = map f
  where
    f (i, o) = sumElements $ rss (run net i) o

main = do
  gen <- newIOGenM $ mkStdGen 10
  net <- randNet [2, 4, 1] gen
  let trainingData = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])] :: [(Vector R, Vector R)]
  let net' = train net 0.5 . take 7000 . cycle $ trainingData
  -- let as = activations net input
  -- let net' = fit net 0.2 input output as
  print $ test net' trainingData
  print $ map (run net' . fst) trainingData
