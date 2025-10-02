from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Index
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime, timedelta

DATABASE_URL = "sqlite:///./vehicles.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class VehicleLog(Base):
    __tablename__ = "vehicle_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    vehicle_number = Column(String(20), index=True, nullable=False)
    driver_name = Column(String(100), nullable=True)
    vehicle_type = Column(String(50), nullable=True)
    entry_time = Column(DateTime, default=datetime.now, nullable=False)
    exit_time = Column(DateTime, default=None, nullable=True)
    status = Column(String(10), default="ENTRY")
    operator_id = Column(String(50), default="system")
    image_path = Column(Text, nullable=True)
    gate_id = Column(String(50), default="main_gate")
    remarks = Column(Text, default="")
    
    __table_args__ = (
        Index('idx_entry_time', 'entry_time'),
        Index('idx_exit_time', 'exit_time'),
        Index('idx_vehicle_exit', 'vehicle_number', 'exit_time'),
    )
    
    def __repr__(self):
        return f"<VehicleLog(id={self.id}, vehicle={self.vehicle_number}, status={self.status})>"

class BannedVehicle(Base):
    __tablename__ = "banned_vehicles"
    
    id = Column(Integer, primary_key=True, index=True)
    vehicle_number = Column(String(20), index=True, nullable=False, unique=True)
    reason = Column(Text, nullable=True)
    banned_at = Column(DateTime, default=datetime.now, nullable=False)
    banned_by = Column(String(50), default="admin")
    
    def __repr__(self):
        return f"<BannedVehicle(id={self.id}, vehicle={self.vehicle_number})>"

# Create tables
Base.metadata.create_all(bind=engine)

def prune_old_records(db: Session, days: int = 30):
    """Remove vehicle logs older than specified days."""
    cutoff = datetime.now() - timedelta(days=days)
    db.query(VehicleLog).filter(VehicleLog.entry_time < cutoff).delete()
    db.commit()