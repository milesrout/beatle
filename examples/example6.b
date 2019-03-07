#a = 0
#b = 1
#c = 3
#zero_result = 0
#nonzero_result = 1
#
#def foo(x):
#    yield from x
#    print(str(x))
#    pass
#
#if a == 0:
#    if b == 0:
#        if c == 0:
#            pass
#            #(print("hello"),
#            # print("hello"))
#        print("goodbye")
#    print("world")
#print("okay")
#
#
#if a == 0 or b == 1:
#    a = (if a == 0:
#             zero_result
#         else:
#             nonzero_result)
#    b = (if b == 0:
#             zero_result
#         else:
#             nonzero_result)
#    print(str(a), str(b))
#
## we don't do macroexpansion yet - need to do it before we do
## type-checking unless I use structured binding types (λ_s calculus) to
## have type-checked hygenic macros.
## 2019: why did I ever think that quasiquoting had anything to do with macros??
#
x = (
        1,
        2,
        3
    )

# You cannot reference local variables with eval
# print(str(eval(\x)))

#print(str(eval(\[$x])))
#print(str([x]))
#
#print(str(eval(\{1, 2, 3})))
#print(str(\${1, 2, 3}))
# 
# 
#print(str((def(a):
#               confuser = (
#                   def(b = (yield)):
#                       yield from b
#               )
#               return None)))
#

def f(a):
   return

# def foo():
#     print_hello = (def(a, b, c):
#                        print(str(def():
#                                    print()
#                                    print(
#                                       str((1, 2, 3)),
#                                     str((a, b, c)))))
#                        print('Hello',
#                              str((def():
#                                      print('1', '2', '3'); print(a, b, c)
#                                      pass)),
#                         'World'))
# print(str(foo))
