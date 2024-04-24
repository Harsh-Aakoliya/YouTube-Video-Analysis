from rhinorefine import Preprocessor
preprocessor = Preprocessor('file.csv')



# TODO: Fill with mean, median and mode 
# preprocessor.fillwithmean("age")
# preprocessor.fillwithmedian("age")
# preprocessor.fillwithmode("age")


#TODO: Removing columns
# preprocessor.removeColumn("age")


#TODO: finding null values
#so it will print all the null values from giving csv file
# print(preprocessor.nullValues())

#TODO: standerisation all the columns
# preprocessor.standardizeColumn(["age"])
# preprocessor.standardizeData()


#TODO: it will normalize all the columns between [0,1]
# preprocessor.normalizeColumn(['age'])
# print(preprocessor.normalizeData())


#TODO: categorical encoding i.e. onehot encoding
# preprocessor.categoricalEncoding("gender")



preprocessor1=Preprocessor("temp.csv")
#TODO: K means clustering algorithm. no of clusters we have to pass as an argument
# print(preprocessor1.compressLossy(5))
 
#TODO: PCA as an argument we have to pass no of dimension in which we have to reduce
# print(preprocessor1.compressNonLossy(1))


#finally for saving processed csv file
preprocessor.save()