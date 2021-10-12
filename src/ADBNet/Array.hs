-- | Array Extensions

module ADBNet.Array
    ( azip
    , alast
    , ahead
    ) where

import           Data.Array.Base
import           Data.Ix

{-# INLINE azip #-}
azip
    :: (IArray a e'', IArray a e', IArray a e, Ix i)
    => (e'' -> e' -> e)
    -> a i e''
    -> a i e'
    -> a i e
azip f x y =
    let (n, b) = if nx > ny then (ny, bounds y) else (nx, bounds x)
    in  unsafeArray
            b
            [ (i, f (unsafeAt x i) (unsafeAt y i)) | i <- [0 .. n - 1] ]
  where
    nx = numElements x
    ny = numElements y

{-# INLINE ahead #-}
ahead :: (IArray a e, Ix i) => a i e -> e
ahead arr = arr ! fst (bounds arr)

{-# INLINE alast #-}
alast :: (IArray a e, Ix i) => a i e -> e
alast arr = arr ! snd (bounds arr)
