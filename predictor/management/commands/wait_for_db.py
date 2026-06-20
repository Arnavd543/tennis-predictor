import time
from django.db import connection
from django.db.utils import OperationalError
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Wait until the database is ready to accept connections.'

    def handle(self, *args, **options):
        self.stdout.write('Waiting for database...')
        for attempt in range(30):
            try:
                connection.ensure_connection()
                self.stdout.write(self.style.SUCCESS('Database ready.'))
                return
            except OperationalError:
                self.stdout.write(f'  Not ready yet (attempt {attempt + 1}/30), retrying in 2s...')
                time.sleep(2)
        raise Exception('Database not available after 60 seconds.')
