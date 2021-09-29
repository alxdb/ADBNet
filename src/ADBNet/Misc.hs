module ADBNet.Misc where

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: Floating a => a -> a
sigmoid' x = sigmoid x * (1 - sigmoid x)

rss :: Floating a => a -> a -> a
rss x y = (x - y) ** 2

rss' :: Floating a => a -> a -> a
rss' x y = 2 * (x - y)
