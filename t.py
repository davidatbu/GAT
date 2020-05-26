import pydot

g = pydot.Dot(root='"a"')
print(g.to_string())
g.write_png("out.png", prog="twopi")
