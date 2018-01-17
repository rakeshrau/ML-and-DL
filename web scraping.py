from bs4 import BeautifulSoup
import urllib.request
import pandas as pd

stock_data = pd.DataFrame('Company','Expiry Date','Price')
    
x = urllib.request.urlopen('http://www.moneycontrol.com/stocks/fno/marketstats/futures/gainers/homebody.php?opttopic=&optinst=allfut&sel_mth=all&sort_order=0').read()
soup =BeautifulSoup(x,'html.parser')

#This gets the name of the stocks from Moneycontrol Top Gainers
##<td class = "TAL">
##  <a href = "some url" class = "b1_12  title = "name of Company"> 
##</a>
##</td>
##<td> Expiry Date </td>     --> 0th Sibling
##<td> Stock Price </td>  --> 1st sibling
td = soup.find_all('td',attrs={'class': 'TAL'})   ##Find all td
l1 =  []
l2 = []
l3 = []
for tds in td:
    l1.append(tds.find('a').get('title'))   ##Get the name of company
    l2.append(tds.findNextSiblings()[0].get_text())    ##Get the Expiry date
    l3.append(tds.findNextSiblings()[1].get_text())    ##Get the stock price 

stock_data['Company']= l1
stock_data['Expiry Date'] =l2
stock_data['Price'] =l3
          
##Generate Excel from Dataframe
writer = pd.ExcelWriter('C:\\Users\\rraushan\\Desktop\\New Stuff\\Udemy ML\\Dataset\\stockprice.xlsx',engine='xlsxwriter')
stock_data.to_excel(writer,index=False, sheet_name='Sheet1')
writer.save()

import pymysql.cursors
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='rakssneha123',
                             db='test',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
print('Trying to do data entry')
try:
    with connection.cursor() as cursor:
        # Create a new record
        for i in range(len(stock_data)):
            sql = "INSERT INTO `stocks` (`Company`, `Expiry_Date`, `Price`) VALUES (%s, %s,%s )"
            cursor.execute(sql, (l1[i],l2[i],l3[i]))
        #cursor.execute(sql, ('Cisco','28-Mar-18','345.23'))
    connection.commit()

    with connection.cursor() as cursor:
        sql = "SELECT `Expiry_Date`, `Price` FROM `stocks` WHERE `Company` = %s"
        cursor.execute(sql, ('BHEL',))
        result  =cursor.fetchone()
        print(result)
        
except:
    print('Failed to do  Database Querry')
    
    
finally:
    connection.close()

