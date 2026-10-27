-- Q1
SELECT customer_id, SUM(order_amount) AS qualifying_spend FROM orders WHERE status='completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000 ORDER BY customer_id;

-- Q2
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;

-- Q3
SELECT product_id, product_name, price_inr FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4
SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5
WITH customer_spend AS (SELECT customer_id, SUM(order_amount) AS total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) AS avg_total FROM customer_spend) SELECT cs.customer_id, cs.total_spend FROM customer_spend cs CROSS JOIN avg_spend a WHERE cs.total_spend>a.avg_total ORDER BY cs.customer_id;

-- Q6
WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM customers c JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT customer_id,city,total_spend,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) AS row_num FROM spend ORDER BY city,spend_rank,customer_id;

-- Q7
WITH daily AS (SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,ROUND(revenue,2) revenue,ROUND(SUM(revenue) OVER(ORDER BY day),2) cumulative_revenue FROM daily ORDER BY day;

