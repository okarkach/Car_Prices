library(tibble)
library(lubridate)
library(dplyr)
library(rvest)
library(stringr)
library(tidyr)
library(rebus)
library(reshape2)


i<-1:50
carsdata<-data.frame()
for (i in 1:50){
  
  html<-read_html(str_c("https://www.cars.com/for-sale/searchresults.action/?page=",i,"&perPage=100&rd=100&searchSource=PAGINATION&shippable-dealers-checkbox=true&showMore=true&sort=relevance&zc=03824&localVehicles=false"),encoding = 'UTF-8')
  get_price<-function(html){
    html%>%
      html_nodes('.listing-row__price')%>%
      html_text('span')%>%
      str_trim()%>%
      unlist()
  }
  get_mileage<-function(html){
    html%>%
      html_nodes('.listing-row__details')%>%
      html_node('.listing-row__mileage')%>%
      html_text('span')%>%
      str_trim()%>%
      unlist()
  }
  get_title<-function(html){
    html%>%
      html_nodes('.listing-row__title')%>%
      html_text('span')%>%
      str_trim()%>%
      unlist()
  }
  get_ratings<-function(html){
    html%>%
      html_nodes('.listing-row__dealer')%>%
      html_node('.dealer-rating-stars')%>%
      html_text('span')%>%
      str_trim()%>%
      unlist()
  }
  price<-data.frame(get_price(html))
  mileage<-data.frame(get_mileage(html))
  title<- data.frame(get_title(html))
  ratings<- data.frame(get_ratings(html))
  
  
 carsdata<-rbind(carsdata,data.frame(title,price,mileage,ratings))
}

carsdata1<- carsdata
colnames(carsdata1)<- c('title','price','mileage',"ratings", 5:ncol(carsdata1))

#Data Cleansing
#Split title column into year , make, model
cars_sep<- carsdata1 %>% separate(title, c('year','make','model'), remove= TRUE, convert=TRUE, extra='drop',fill='left')

#Removing $ and ',' from price
cars_sep$price <- str_replace_all(cars_sep$price, "[^[:alnum:]]", "")

#Removing 'mi' and ',' from mileage
cars_sep$mileage<- str_replace_all(cars_sep$mileage, "[^[:alnum:]]", "")

cars_sep$mileage<- str_remove_all(cars_sep$mileage, pattern = "mi")
#Cleaning ratings
cars_sep$ratings <- str_remove_all(cars_sep$ratings, pattern = "(" %R% one_or_more(DGT) %R% " " %R% "reviews" %R% ")")
cars_sep$ratings <- gsub("[^[:digit:]., ]", "", cars_sep$ratings)

library(rebus)
example_str <- "This is 3345 so cool 345"
pattern = one_or_more(DGT) %R% END 
