module ADBNet.Network
    ( NeuralNetwork
    , networkDims
    , randomNetworkM
    , randomNetwork
    , randomNetwork_
    , activations
    , deltas
    , gradients
    ) where

import           ADBNet.Misc
import           ADBNet.Tensor
import           Data.Array
import           System.Random.Stateful

type M = Matrix Double
type V = Vector Double
type A = Array Int

type NeuralNetwork = A M

networkDims :: [Int] -> [(Int, Int)]
networkDims d = zipWith (\i o -> (o, i + 1)) d (tail d)

randomNetworkM :: (StatefulGen g m) => [Int] -> g -> m NeuralNetwork
randomNetworkM d g = listArray (1, length d - 1)
    <$> mapM (\i -> trndM i (0.0, 1.0) g) (networkDims d)

randomNetwork :: (RandomGen g) => [Int] -> g -> (NeuralNetwork, g)
randomNetwork d g = runStateGen g (randomNetworkM d)

randomNetwork_ :: [Int] -> Int -> NeuralNetwork
randomNetwork_ d g = runStateGen_ (mkStdGen g) (randomNetworkM d)

-- Central Functions

activations :: NeuralNetwork -> V -> (A V, A V)
activations nn i = (as, zs)  where
    s  = snd $ bounds nn
    as = listArray (1, s) [ tmap sigmoid $ z i | i <- [1 .. s] ]
    zs = listArray (1, s) [ z i | i <- [1 .. s] ]
    z n = matmul (nn ! n) (vapp l 1.0)
      where
        l | n == 1    = i
          | otherwise = as ! (n - 1)

deltas :: NeuralNetwork -> (A V, A V) -> V -> A V
deltas nn (as, zs) o = ds  where
    s  = snd $ bounds nn
    ds = listArray (1, s) [ tmap sigmoid' (zs ! i) * d i | i <- [1, s] ]
    d n | n == s    = tzip rss' (as ! s) o
        | otherwise = transp (remCol (nn ! (n + 1))) `matmul` (ds ! (n + 1))

gradients :: NeuralNetwork -> V -> (A V, A V) -> A V -> A M
gradients nn i (as, zs) ds = gs  where
    s  = snd $ bounds nn
    gs = listArray (1, s) [ g i | i <- [1 .. s] ]
    g n = (ds ! n) `outerp` vapp l 1.0
      where
        l | n == 1    = i
          | otherwise = as ! (n - 1)
