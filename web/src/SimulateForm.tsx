import { Form, FormField, FormLabel } from '@radix-ui/react-form';
import { Button, Card, Flex, Heading, Separator, TextField } from '@radix-ui/themes';
import _ from 'lodash';
import React, { useCallback, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Routes } from 'routes';

type FormValue = number | '';
interface FormData {
  Body1: {
    position: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    velocity: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    mass: FormValue;
  };
  Body2: {
    position: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    velocity: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    mass: FormValue;
  };
}

const SimulateForm: React.FC = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState<FormData>({
    Body1: { position: {x: -0.73, y: 0, z: 0}, velocity: {x: 0, y: -0.0015, z: 0}, mass: 1 },
    Body2: { position: {x: 60.34, y: 0, z: 0}, velocity: {x: 0, y: 0, z: 0.13}, mass: 0.123 },
  });

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    let newValue: FormValue = value === '' ? '' : parseFloat(value);
    setFormData((prev) => _.set({ ...prev }, name, newValue));
  }, []);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      try {
        const response = await fetch('http://localhost:8000/simulation', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
        });
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        navigate(Routes.SIMULATION);
      } catch (error) {
        console.error('Error:', error);
      }
    },
    [formData]
  );

  return (
    <div
      style={{
        position: 'absolute',
        top: '5%',
        left: 'calc(50% - 200px)',
        overflow: 'scroll',
      }}
    >
      {/* Card: https://www.radix-ui.com/themes/docs/components/card */}
      <Card
        style={{
          width: '400px',
        }}
      >
        <Heading as="h2" size="4" weight="bold" mb="4">
          Run a Simulation
        </Heading>
        <Link to={Routes.SIMULATION}>View previous simulation</Link>
        <Separator size="4" my="5" />
        <Form onSubmit={handleSubmit}>
          {/* 
            *********************************
            Body1
            *********************************
            */}
          <Heading as="h3" size="3" weight="bold">
            Body1
          </Heading>
          {/* Form: https://www.radix-ui.com/primitives/docs/components/form */}
          <FormField name="Body1.position.x">
            <FormLabel htmlFor="Body1.position.x">Initial X-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.position.x"
              name="Body1.position.x"
              value={formData.Body1.position.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.position.y">
            <FormLabel htmlFor="Body1.position.y">Initial Y-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.position.y"
              name="Body1.position.y"
              value={formData.Body1.position.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.position.z">
            <FormLabel htmlFor="Body1.position.z">Initial Z-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.position.z"
              name="Body1.position.z"
              value={formData.Body1.position.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.velocity.x">
            <FormLabel htmlFor="Body1.velocity.x">Initial X-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.velocity.x"
              name="Body1.velocity.x"
              value={formData.Body1.velocity.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.velocity.y">
            <FormLabel htmlFor="Body1.velocity.y">Initial Y-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.velocity.y"
              name="Body1.velocity.y"
              value={formData.Body1.velocity.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.velocity.z">
            <FormLabel htmlFor="Body1.velocity.z">Initial Z-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.velocity.z"
              name="Body1.velocity.z"
              value={formData.Body1.velocity.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.mass">
            <FormLabel htmlFor="Body1.mass">Mass</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.mass"
              name="Body1.mass"
              value={formData.Body1.mass}
              onChange={handleChange}
              required
            />
          </FormField>
          {/* 
            *********************************
            Body2
            *********************************
             */}
          <Heading as="h3" size="3" weight="bold" mt="4">
            Body2
          </Heading>
          <FormField name="Body2.position.x">
            <FormLabel htmlFor="Body2.position.x">Initial X-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.position.x"
              name="Body2.position.x"
              value={formData.Body2.position.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.position.y">
            <FormLabel htmlFor="Body2.position.y">Initial Y-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.position.y"
              name="Body2.position.y"
              value={formData.Body2.position.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.position.z">
            <FormLabel htmlFor="Body2.position.z">Initial Z-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.position.z"
              name="Body2.position.z"
              value={formData.Body2.position.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.velocity.x">
            <FormLabel htmlFor="Body2.velocity.x">Initial X-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.velocity.x"
              name="Body2.velocity.x"
              value={formData.Body2.velocity.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.velocity.y">
            <FormLabel htmlFor="Body2.velocity.y">Initial Y-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.velocity.y"
              name="Body2.velocity.y"
              value={formData.Body2.velocity.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.velocity.z">
            <FormLabel htmlFor="Body2.velocity.z">Initial Z-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.velocity.z"
              name="Body2.velocity.z"
              value={formData.Body2.velocity.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.mass">
            <FormLabel htmlFor="Body2.mass">Mass</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.mass"
              name="Body2.mass"
              value={formData.Body2.mass}
              onChange={handleChange}
              required
            />
          </FormField>
          <Flex justify="center" m="5">
            <Button type="submit">Submit</Button>
          </Flex>
        </Form>
      </Card>
    </div>
  );
};

export default SimulateForm;
