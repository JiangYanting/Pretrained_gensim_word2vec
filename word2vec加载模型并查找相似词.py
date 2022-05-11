from gensim.models import word2vec
import logging
import gensim



# 对应的加载方式
#model = gensim.models.KeyedVectors.load_word2vec_format("model(单独训练).bin",binary=True)
model =word2vec.Word2Vec.load("词向量模型.model")


# 功能1：计算某个词的相关词列表
# 计算某个词的相关词列表

#y2 = model.most_similar("西红柿/n", topn=10)  # 8个最相关的。
#print ("和【西红柿】最相关的词有：\n")
#for item in y2:
#    print(item[0], item[1])
#print("\n")
y2 = model.most_similar("book", topn=10)  # 8个最相关的。
print ("和【book】最相关的词有：\n")
for item in y2:
    print(item[0], item[1])
print("\n") 




# 功能2：找出某个不合群的词
#b1 = model.doesnt_match("番犇 次犇 场犇 回犇 遍犇 匝犇 遭犇 下犇 通犇 周犇 顿犇 通犇".split(" "))  #寻找不合群的词
#print("动量词中，不合群的词：",b1,"\n")

print("\n\n\n")
# 功能3：计算两个词的相似度/相关程度
#c1  = model.similarity(u"以", u"凭")
#print("【以】和【凭】的相似度为：", c1)
#print("\n\n\n")

#c2  = model.similarity(u"以", u"从")
#print("【以】和【从】的相似度为：", c2)

#print("\n\n\n")

# 功能4：获取某个词的词向量
#print (model['就'])
#print(type(model['就']))
#print("\n\n\n")

#功能5 ：analogy类比推理

print("妈妈-爸爸 ≈ 女人 - ？\n")
e1 = model.most_similar([u'女人',u'爸爸'], ['妈妈'], topn=10)    #走 一 遭 = 去 一 ？
for item in e1:
    print(item[0], item[1])


#print("\n\n\n")
print("莫斯科-北京 ≈ 俄罗斯 - ？\n")
e1 = model.most_similar([u'俄罗斯',u'北京'], ['莫斯科'], topn=10)    #走 一 趟 = ？ 一 遭
for item in e1:
    print(item[0], item[1])

print("\n\n\n")

    
print("江苏-四川 ≈ 南京 - ？\n")
e2 = model.most_similar(positive=[u'南京', u'四川'], negative=[u'江苏'], topn=10)   #   皇帝 - 谕旨 = 丞相 - ？  
for item in e2:
    print(item[0], item[1])
    
