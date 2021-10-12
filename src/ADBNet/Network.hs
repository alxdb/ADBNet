module ADBNet.Network
  ( NeuralNetwork
  , TrainingParameters(TrainingParameters)
  , learningRate
  , trainingIterations
  , networkDims
  , randomNetworkM
  , randomNetwork
  , randomNetwork_
  , activations
  , deltas
  , gradients
  , train
  , run
  ) where

import           ADBNet.Array
import           ADBNet.Misc
import           ADBNet.Tensor
import           Control.Monad
import           Control.Monad.ST
import           Data.Array.IArray
import           Data.STRef
import           Debug.Trace
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
activations nn i = (as, zs)
 where
  s  = snd $ bounds nn
  as = listArray (1, s) [ tmap sigmoid $ z i | i <- [1 .. s] ]
  zs = listArray (1, s) [ z i | i <- [1 .. s] ]
  z n = matmul (nn ! n) (vapp l 1.0)
   where
    l | n == 1    = i
      | otherwise = as ! (n - 1)

deltas :: NeuralNetwork -> (A V, A V) -> V -> A V
deltas nn (as, zs) o = ds
 where
  s  = snd $ bounds nn
  ds = listArray (1, s) [ tmap sigmoid' (zs ! i) * d i | i <- [1, s] ]
  d n | n == s    = tzip rss' (as ! s) o
      | otherwise = transp (remCol (nn ! (n + 1))) `matmul` (ds ! (n + 1))

gradients :: NeuralNetwork -> V -> (A V, A V) -> A V -> A M
gradients nn i (as, zs) ds = gs
 where
  s  = snd $ bounds nn
  gs = listArray (1, s) [ g i | i <- [1 .. s] ]
  g n = (ds ! n) `outerp` vapp l 1.0
   where
    l | n == 1    = i
      | otherwise = as ! (n - 1)

-- Training

data TrainingParameters = TrainingParameters
  { learningRate       :: Double
  , trainingIterations :: Int
  }

type TrainingMeta = A (Double, V)

type TrainingData = [(V, V)]

loss :: V -> V -> Double
loss x y = let d = (x - y) in dotv d d

singleFit :: Double -> NeuralNetwork -> (V, V) -> (NeuralNetwork, Double, V)
singleFit lr nn (i, o) = traceShow (nn', loss o' o, o') (nn', loss o' o, o')
 where
  o'  = alast $ fst az
  az  = traceShow az $ activations nn i
  ds  = trace "del" $ deltas nn az o
  gs  = trace "gra" $ gradients nn o az ds
  nn' = trace "app" $ azip (-) nn $ amap (scale lr) gs

train
  :: TrainingParameters
  -> NeuralNetwork
  -> TrainingData
  -> (NeuralNetwork, TrainingMeta)
train tp nn td = runST $ do
  ref <- newSTRef (nn, listArray (1, ti) $ repeat (0, tval outDim 0))
  forM_ (zip [1 .. ti] (cycle td)) $ f ref
  readSTRef ref
 where
  lr = learningRate tp
  ti = trainingIterations tp
  f :: STRef s (NeuralNetwork, TrainingMeta) -> (Int, (V, V)) -> ST s ()
  f ref (n, trainingExample) = traceShow n $ modifySTRef ref $ \(nn, tm) ->
    let (nn', l, a) = traceShow nn $ singleFit lr nn trainingExample
    in  (nn', tm // [(n, (l, a))])
  outDim = dims . snd . head $ td

-- Running

run :: NeuralNetwork -> V -> V
run nn i = alast . fst $ activations nn i
