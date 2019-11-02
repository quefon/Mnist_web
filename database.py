import redis

pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)
#r.set('name', 'quefon')
r.setnx('name', 'quefon')
print(r['name'])
print(r.get('name'))
print(type(r.get('name')))
